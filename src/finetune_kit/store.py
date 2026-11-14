"""Run directories, request files, results.tsv and run summaries."""
import json
import os
import re
import statistics
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Literal

import finetune_kit
from finetune_kit.experiment import ExperimentConfig, MetricRef, State, load
from finetune_kit.profiles import ModelProfile, load_profiles
from finetune_kit.stats import RunSummary
from finetune_kit.train.config import defaults_for, nondefault_count

Stage = Literal["bakeoff", "lr-scan", "noise", "research", "promote"]
TERMINAL = ("done", "crash", "timeout")
RESULT_COLUMNS = [
    "run_id", "round", "stage", "objective_mean", "objective_std", "guardrail_mean",
    "n_seeds", "minutes", "verdict", "description", "config",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, obj: dict) -> None:
    """Atomic write: readers (e.g. `finetune-kit wait`) never see a partial file."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def project_root(exp_dir: Path) -> Path:
    return Path(exp_dir).resolve().parent.parent


def run_dir(exp_dir: Path, run_id: str) -> Path:
    return Path(exp_dir) / "runs" / run_id


def read_request(exp_dir: Path, run_id: str) -> dict:
    p = run_dir(exp_dir, run_id) / "request.json"
    if not p.is_file():
        raise FileNotFoundError(f"no run {run_id!r} in {Path(exp_dir).name}")
    return json.loads(p.read_text())


def write_request(exp_dir: Path, run_id: str, req: dict) -> None:
    write_json(run_dir(exp_dir, run_id) / "request.json", req)


def list_requests(exp_dir: Path) -> list[dict]:
    runs = Path(exp_dir) / "runs"
    if not runs.is_dir():
        return []
    out = []
    for d in sorted(runs.iterdir()):
        p = d / "request.json"
        if p.is_file():
            out.append(json.loads(p.read_text()))
    return out


def new_run(
    exp_dir: Path, stage: Stage, overrides: dict, seeds: list[int], description: str,
    *, extra: dict | None = None,
) -> str:
    exp_dir = Path(exp_dir)
    runs = exp_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    _, _, state = load(exp_dir)
    while True:
        nums = [int(m.group(1)) for d in runs.iterdir() if (m := re.fullmatch(r"r(\d+)", d.name))]
        run_id = f"r{max(nums, default=0) + 1:03d}"
        try:
            (runs / run_id).mkdir()
            break
        except FileExistsError:
            continue
    req = {
        "run_id": run_id, "stage": stage, "round": state.round, "overrides": dict(overrides),
        "seeds": list(seeds), "description": description, "status": "queued",
        "created": now(), "retries": 0, **(extra or {}),
    }
    write_request(exp_dir, run_id, req)
    return run_id


def profile_for(cfg: ExperimentConfig, overrides: dict, root: Path) -> ModelProfile:
    key = overrides.get("model", cfg.model)
    profiles = load_profiles(root)
    if key not in profiles:
        raise ValueError(f"unknown model profile {key!r}")
    return profiles[key]


def effective_defaults(cfg: ExperimentConfig, state: State, profile: ModelProfile) -> dict:
    """Profile defaults for the task, with the LR-scan result replacing the profile LR."""
    d = defaults_for(profile, cfg.task)
    if state.lr_scan_best is not None and profile.key == cfg.model:
        d["lr"] = state.lr_scan_best
    return d


def seed_metrics(exp_dir: Path, run_id: str) -> list[dict]:
    """metrics.json of every seed that produced one, in request seed order."""
    req = read_request(exp_dir, run_id)
    out = []
    for s in req["seeds"]:
        p = run_dir(exp_dir, run_id) / f"seed_{s}" / "metrics.json"
        if p.is_file():
            out.append({"seed": s, **json.loads(p.read_text())})
    return out


def summarize(exp_dir: Path, run_id: str) -> RunSummary:
    exp_dir = Path(exp_dir)
    cfg, _, state = load(exp_dir)
    req = read_request(exp_dir, run_id)
    ms = [m for m in seed_metrics(exp_dir, run_id) if m.get("objective") is not None]
    if not ms:
        raise ValueError(f"run {run_id} has no finished seeds")
    objs = [float(m["objective"]) for m in ms]
    guards = [float(m["guardrail"]) for m in ms if m.get("guardrail") is not None]
    overrides = req.get("overrides") or {}
    profile = profile_for(cfg, overrides, project_root(exp_dir))
    tunables = {k: v for k, v in overrides.items() if k not in ("model", "step_fraction")}
    return RunSummary(
        objective_mean=statistics.fmean(objs),
        objective_std=statistics.stdev(objs) if len(objs) > 1 else 0.0,
        guardrail_mean=statistics.fmean(guards) if guards else None,
        n_seeds=len(objs),
        n_nondefault=nondefault_count(tunables, effective_defaults(cfg, state, profile)),
    )


def metric_info(cfg: ExperimentConfig, ref: MetricRef) -> tuple[str, float]:
    """(direction, min_delta) of a metric reference, custom metrics included."""
    cm = cfg.custom_metric
    if cm is not None and ref.metric == cm.func:
        return cm.direction, 0.002
    from finetune_kit import metrics  # lazy: pulls in scikit-learn

    m = metrics.get(ref.metric)
    return m.direction, m.min_delta


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return re.sub(r"[\t\r\n]+", " ", str(v))


def append_result(exp_dir: Path, row: dict) -> None:
    p = Path(exp_dir) / "results.tsv"
    lines = [] if p.is_file() else ["\t".join(RESULT_COLUMNS)]
    lines.append("\t".join(_cell(row.get(c)) for c in RESULT_COLUMNS))
    with p.open("a") as f:
        f.write("\n".join(lines) + "\n")


def result_run_ids(exp_dir: Path) -> set[str]:
    p = Path(exp_dir) / "results.tsv"
    if not p.is_file():
        return set()
    return {line.split("\t", 1)[0] for line in p.read_text().splitlines()[1:] if line}


def current_versions(profile: ModelProfile) -> dict[str, str]:
    out = {"finetune-kit": finetune_kit.__version__, "model_revision": profile.revision}
    for pkg in ("torch", "transformers"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def version_warnings(cfg: ExperimentConfig, profile: ModelProfile) -> list[str]:
    cur = current_versions(profile)
    return [
        f"{k} is {cur[k]} but this experiment recorded {v}; results may not be comparable"
        for k, v in cfg.versions.items() if k in cur and cur[k] != v
    ]
