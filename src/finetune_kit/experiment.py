import fcntl
import hashlib
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from finetune_kit import catalog

SPACE_FILE = "search-space.yaml"


class TamperError(Exception):
    pass


class DataPaths(BaseModel):
    train: str
    val: str | None = None
    test: str | None = None


class MetricRef(BaseModel):
    metric: str
    params: dict = Field(default_factory=dict)


class CustomMetric(BaseModel):
    path: str
    func: str
    direction: str


class Budget(BaseModel):
    max_runs: int = 40
    max_hours: float = 24


class ExperimentConfig(BaseModel):
    uid: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    task: Literal["seq-cls", "token-cls"]
    model: str
    data: DataPaths
    objective: MetricRef
    guardrail: MetricRef | None = None
    positive_class: str | None = None
    custom_metric: CustomMetric | None = None
    seeds: int = 3
    budget: Budget = Field(default_factory=Budget)
    research_subset: int | None = None
    min_delta: float | None = None
    versions: dict[str, str] = Field(default_factory=dict)


class SearchSpace(BaseModel):
    round: int
    tunable: list[str]
    losses: list[str]


class State(BaseModel):
    round: int = 0
    checksums: dict[str, str] = Field(default_factory=dict)
    sigma_obj: float | None = None
    sigma_guard: float | None = None
    noise_run: str | None = None
    best_run: str | None = None
    lr_scan_best: float | None = None
    baseline_train_s: float | None = None
    non_keep_streak: int = 0
    runs_used: int = 0
    hours_used: float = 0
    promoted: bool = False
    bakeoff: dict | None = None
    model_locked: bool = False
    finalized: dict[str, str] = Field(default_factory=dict)  # run_id -> verdict, written by the worker
    request_hashes: dict[str, str] = Field(default_factory=dict)  # run_id -> sha256 at enqueue


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _dump_yaml(path: Path, model: BaseModel) -> None:
    path.write_text(yaml.safe_dump(model.model_dump(mode="json"), sort_keys=False))


def new_experiment(root: Path, name: str, cfg: ExperimentConfig) -> Path:
    exp_root = Path(root) / "experiments"
    exp_root.mkdir(parents=True, exist_ok=True)
    nums = [int(m.group(1)) for d in exp_root.iterdir() if (m := re.match(r"(\d+)-", d.name))]
    d = exp_root / f"{max(nums, default=0) + 1:03d}-{_slug(name)}"
    d.mkdir()
    _dump_yaml(d / "experiment.yaml", cfg)
    save_state(d, State())
    return d


def load(exp_dir: Path) -> tuple[ExperimentConfig, SearchSpace | None, State]:
    exp_dir = Path(exp_dir)
    cfg = ExperimentConfig.model_validate(yaml.safe_load((exp_dir / "experiment.yaml").read_text()))
    space_path = exp_dir / SPACE_FILE
    space = (
        SearchSpace.model_validate(yaml.safe_load(space_path.read_text()))
        if space_path.is_file() else None
    )
    state = State.model_validate_json((exp_dir / "state.json").read_text())
    return cfg, space, state


def save_state(exp_dir: Path, state: State) -> None:
    """Atomic write (temp file + os.replace): readers never see a torn state.json."""
    path = Path(exp_dir) / "state.json"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".state.")
    with os.fdopen(fd, "w") as f:
        f.write(state.model_dump_json(indent=2))
    os.replace(tmp, path)


_held: dict[str, int] = {}


@contextmanager
def state_lock(exp_dir: Path):
    """Per-experiment exclusive lock (`<exp>/.state.lock`) around load-modify-save of state.

    Re-entrant within a process, so library helpers can lock while their callers already do.
    """
    key = str(Path(exp_dir).resolve())
    if _held.get(key):
        _held[key] += 1
        try:
            yield
        finally:
            _held[key] -= 1
        return
    with open(Path(key) / ".state.lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        _held[key] = 1
        try:
            yield
        finally:
            _held.pop(key, None)
            fcntl.flock(f, fcntl.LOCK_UN)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commit_space(exp_dir: Path, space: SearchSpace) -> State:
    exp_dir = Path(exp_dir)
    with state_lock(exp_dir):
        return _commit_space(exp_dir, space)


def _commit_space(exp_dir: Path, space: SearchSpace) -> State:
    cfg, _, state = load(exp_dir)
    valid = set(catalog.names(cfg.task))
    for n in space.tunable:
        if n not in valid:
            raise ValueError(f"unknown tunable hyperparameter {n!r} for {cfg.task}")
    allowed = catalog.LOSSES[cfg.task]
    for l in space.losses:
        if l not in allowed:
            raise ValueError(f"loss {l!r} is not allowed for {cfg.task} (allowed: {', '.join(allowed)})")
    state.round += 1
    _dump_yaml(exp_dir / SPACE_FILE, space.model_copy(update={"round": state.round}))
    state.checksums[SPACE_FILE] = _sha(exp_dir / SPACE_FILE)
    save_state(exp_dir, state)
    return state


def protected_files(exp_dir: Path, project_root: Path) -> list[Path]:
    exp_dir, project_root = Path(exp_dir), Path(project_root)
    return [
        project_root / ".finetune-kit" / "memory" / "research-rules.md",
        exp_dir / SPACE_FILE,
        exp_dir / "splits" / "train.jsonl",
        exp_dir / "splits" / "val.jsonl",
        exp_dir / "splits" / "test.meta.json",
    ]


def _key(p: Path, exp_dir: Path, project_root: Path) -> str:
    base = exp_dir if p.is_relative_to(exp_dir) else project_root
    return p.relative_to(base).as_posix()


def record_checksums(exp_dir: Path, project_root: Path) -> State:
    exp_dir, project_root = Path(exp_dir), Path(project_root)
    with state_lock(exp_dir):
        _, _, state = load(exp_dir)
        for p in protected_files(exp_dir, project_root):
            if p.is_file():
                state.checksums[_key(p, exp_dir, project_root)] = _sha(p)
        save_state(exp_dir, state)
    return state


def verify_checksums(exp_dir: Path, project_root: Path) -> None:
    exp_dir, project_root = Path(exp_dir), Path(project_root)
    _, _, state = load(exp_dir)
    bad = []
    for p in protected_files(exp_dir, project_root):
        key = _key(p, exp_dir, project_root)
        if key not in state.checksums:
            continue
        if not p.is_file():
            bad.append(f"{key} (missing)")
        elif _sha(p) != state.checksums[key]:
            bad.append(f"{key} (changed)")
    if bad:
        raise TamperError("protected files modified: " + ", ".join(bad))
