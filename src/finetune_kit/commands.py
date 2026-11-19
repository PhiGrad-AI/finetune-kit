"""Orchestration behind the setup, data and comparison CLI commands (cli.py keeps thin wrappers)."""
import csv
import json
import math
import re
import shutil
from pathlib import Path

from finetune_kit import data, experiment as ex, metrics, sources, store
from finetune_kit.paths import vault_dir

STREAK_STOP = 10
TEXT_CHARS = 300
LABEL_ISSUE_SCORE = 0.95


def parse_params(raw: str, flag: str) -> dict:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{flag} is not valid JSON: {e}") from None
    if not isinstance(obj, dict):
        raise ValueError(f"{flag} must be a JSON object")
    return obj


def _resolve(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else Path(root) / path


def _metric(cfg: ex.ExperimentConfig, ref: ex.MetricRef, root: Path) -> metrics.Metric:
    cm = cfg.custom_metric
    if cm is not None and ref.metric == cm.func:
        return metrics.load_custom(_resolve(root, cm.path), cm.func, cm.direction)
    return metrics.get(ref.metric)


# ---- new, set data / model / goal ----
# One-shot `new` and the step-by-step `set` commands share these validators (nothing is written until all pass).


def validate_data(
    root: Path, task: str, *, train: str | None, val: str | None, test: str | None,
    source: str | None, revision: str | None, columns: dict | None,
) -> tuple[ex.DataPaths, list[str] | None]:
    """Check a data choice the way check-data will load it; returns the config and seq-cls train labels."""
    ex.check_data_choice(train, source, val, test)
    if (val is None) != (test is None):
        raise ValueError("give both --val and --test, or neither (the kit then splits --train)")
    if revision is not None and not source:
        raise ValueError("--revision only applies with --source")
    if source:
        sources.parse_hf(source)  # one-line error for a malformed spec
    for label, p in (("train", train), ("val", val), ("test", test)):
        if p is not None and not _resolve(root, p).is_file():
            raise FileNotFoundError(f"--{label} file not found: {p}")
    pinned = sources.pin_revision(source, revision) if source else None
    data_cfg = ex.DataPaths(
        train=train, val=val, test=test, source=source, revision=pinned, columns=columns or {}
    )
    # Load every split the way check-data will, so mapping and row errors surface before anything is written.
    got = sources.resolve_source(data_cfg, root, task, 0.1)
    if task != "seq-cls":
        return data_cfg, None
    data.check_label_column(got.train + got.val + got.test)
    train_rows = got.train if source else sources.load_train(data_cfg, root, task)
    return data_cfg, data.label_map(train_rows, task)


def validate_model(root: Path, task: str, model: str, bakeoff: list[str] | None):
    from finetune_kit.profiles import load_profiles

    profiles = load_profiles(root)
    if model not in profiles:
        raise ValueError(f"unknown model {model!r}; see `finetune-kit models`")
    if task not in profiles[model].tasks:
        raise ValueError(f"model {model!r} does not support {task}")
    if bakeoff is not None:
        if not (2 <= len(bakeoff) <= 3) or len(set(bakeoff)) != len(bakeoff) or bakeoff[0] != model or any(
            k not in profiles or task not in profiles[k].tasks for k in bakeoff
        ):
            raise ValueError(f"--bakeoff needs 2-3 models for {task}, the first being --model")
    return profiles[model]


def _check_labels(task: str, labels: list[str] | None, positive_class: str | None, resolved: list) -> None:
    if task != "seq-cls":
        if positive_class is not None:
            raise ValueError("--positive-class applies to seq-cls only")
        return
    if positive_class is not None and positive_class not in labels:
        raise ValueError(f"--positive-class {positive_class!r} is not a train label {labels}")
    if positive_class is None and len(labels) > 2 and any(m.needs_positive for m in resolved):
        names = [m.name for m in resolved if m.needs_positive]
        raise ValueError(
            f"metric {', '.join(names)} needs --positive-class: train has {len(labels)} classes {labels}"
        )


def validate_goal(
    root: Path, task: str, *, metric: str, guardrail: str | None, custom_metric: str | None,
    direction: str | None, positive_class: str | None, labels: list[str] | None,
) -> ex.CustomMetric | None:
    """Check the objective, guardrail, custom metric and positive class; returns the custom metric config."""
    cm = None
    if custom_metric is not None:
        path, sep, func = custom_metric.rpartition(":")
        if not sep or not path or not func:
            raise ValueError("--custom-metric must be PATH:FUNC")
        if direction not in ("max", "min"):
            raise ValueError("--custom-metric needs --direction max|min")
        if task != "seq-cls":
            raise ValueError("custom metrics support seq-cls only")
        metrics.load_custom(_resolve(root, path), func, direction)  # validates file and function
        cm = ex.CustomMetric(path=path, func=func, direction=direction)
    elif direction is not None:
        raise ValueError("--direction only applies with --custom-metric")

    resolved: list[metrics.Metric] = []
    for role, mname in (("objective", metric), ("guardrail", guardrail)):
        if mname is None:
            continue
        if cm is not None and mname == cm.func:
            resolved.append(metrics.load_custom(_resolve(root, cm.path), cm.func, cm.direction))
            continue
        m = metrics.get(mname)
        if task not in m.tasks:
            raise ValueError(f"{role} metric {mname!r} is not valid for {task}")
        resolved.append(m)
    if cm is not None and cm.func not in (metric, guardrail):
        raise ValueError(f"--metric or --guardrail must be the custom metric {cm.func!r}")
    _check_labels(task, labels, positive_class, resolved)
    return cm


def setup_experiment(
    root: Path, name: str, *, task: str, model: str | None = None, train: str | None = None,
    val: str | None = None, test: str | None = None, source: str | None = None,
    revision: str | None = None, metric: str | None = None, metric_params: dict | None = None,
    guardrail: str | None = None, guardrail_params: dict | None = None,
    positive_class: str | None = None, seeds: int = 3, max_runs: int = 40, max_hours: float = 24,
    subset: int | None = None, custom_metric: str | None = None, direction: str | None = None,
    columns: dict | None = None, bakeoff: list[str] | None = None,
) -> Path:
    """Create an experiment. With only `task` it is a draft; data, model and goal come later via `set_*`."""
    root = Path(root)
    if not ex._slug(name):
        raise ValueError(f"experiment name {name!r} has no letters or digits")
    if task not in ("seq-cls", "token-cls"):
        raise ValueError(f"--task must be seq-cls or token-cls, not {task!r}")
    if seeds < 2:
        raise ValueError("--seeds must be at least 2: the noise floor needs a standard deviation")
    budget = ex.Budget(max_runs=max_runs, max_hours=max_hours)
    if model is None and train is None and source is None and metric is None:
        if val or test or revision or guardrail or positive_class or custom_metric or direction or bakeoff or columns:
            raise ValueError("those options need --model, --train or --source, and --metric (or use `set`)")
        cfg = ex.ExperimentConfig(name=name, task=task, seeds=seeds, budget=budget, research_subset=subset)
        return ex.new_experiment(root, name, cfg)

    data_cfg, labels = validate_data(
        root, task, train=train, val=val, test=test, source=source, revision=revision, columns=columns,
    )
    if model is None:
        raise ValueError("give --model")
    profile = validate_model(root, task, model, bakeoff)
    if metric is None:
        raise ValueError("give --metric")
    cm = validate_goal(
        root, task, metric=metric, guardrail=guardrail, custom_metric=custom_metric, direction=direction,
        positive_class=positive_class, labels=labels,
    )
    cfg = ex.ExperimentConfig(
        name=name, task=task, model=model,
        data=data_cfg,
        objective=ex.MetricRef(metric=metric, params=metric_params or {}),
        guardrail=ex.MetricRef(metric=guardrail, params=guardrail_params or {}) if guardrail else None,
        positive_class=positive_class, custom_metric=cm, seeds=seeds,
        budget=budget, research_subset=subset,
        versions=store.current_versions(profile), bakeoff=bakeoff,
    )
    return ex.new_experiment(root, name, cfg)


def _edit_setup(exp_dir: Path, what: str, change) -> ex.ExperimentConfig:
    """Under the state lock: refuse once the data is checked, let `change(cfg, root)` validate and
    return the new config, then rewrite experiment.yaml atomically."""
    exp_dir = Path(exp_dir)
    with ex.state_lock(exp_dir):
        if (exp_dir / "splits" / "train.jsonl").is_file():
            raise ValueError(f"{what} is checked and locked; start a new experiment to change it")
        cfg, _, _ = ex.load(exp_dir)
        cfg = change(cfg, store.project_root(exp_dir))
        ex.save_config(exp_dir, cfg)
    return cfg


def _train_labels(cfg: ex.ExperimentConfig, root: Path) -> list[str] | None:
    if cfg.task != "seq-cls":
        return None
    return data.label_map(sources.load_train(cfg.data, root, cfg.task), cfg.task)


def set_data(
    exp_dir: Path, *, train: str | None = None, val: str | None = None, test: str | None = None,
    source: str | None = None, revision: str | None = None, columns: dict | None = None,
) -> dict:
    def change(cfg, root):
        data_cfg, labels = validate_data(
            root, cfg.task, train=train, val=val, test=test, source=source, revision=revision, columns=columns,
        )
        if cfg.objective is not None:  # the goal was answered first: its labels must still hold
            refs = [r for r in (cfg.objective, cfg.guardrail) if r is not None]
            _check_labels(cfg.task, labels, cfg.positive_class, [_metric(cfg, r, root) for r in refs])
        return cfg.model_copy(update={"data": data_cfg})

    return {"data": _edit_setup(exp_dir, "data", change).data.model_dump()}


def set_model(exp_dir: Path, model: str, compare: list[str] | None = None) -> dict:
    def change(cfg, root):
        bakeoff = list(dict.fromkeys([model, *compare])) if compare else None
        profile = validate_model(root, cfg.task, model, bakeoff)
        return cfg.model_copy(update={
            "model": model, "bakeoff": bakeoff, "versions": store.current_versions(profile),
        })

    cfg = _edit_setup(exp_dir, "model", change)
    return {"model": cfg.model, "bakeoff": cfg.bakeoff}


def set_goal(
    exp_dir: Path, *, metric: str, metric_params: dict | None = None, guardrail: str | None = None,
    guardrail_params: dict | None = None, positive_class: str | None = None,
    custom_metric: str | None = None, direction: str | None = None,
) -> dict:
    def change(cfg, root):
        if cfg.data is None:
            raise ValueError("choose the data first (step: choose-data)")
        cm = validate_goal(
            root, cfg.task, metric=metric, guardrail=guardrail, custom_metric=custom_metric,
            direction=direction, positive_class=positive_class, labels=_train_labels(cfg, root),
        )
        return cfg.model_copy(update={
            "objective": ex.MetricRef(metric=metric, params=metric_params or {}),
            "guardrail": ex.MetricRef(metric=guardrail, params=guardrail_params or {}) if guardrail else None,
            "positive_class": positive_class, "custom_metric": cm,
        })

    cfg = _edit_setup(exp_dir, "goal", change)
    return {
        "objective": cfg.objective.model_dump(),
        "guardrail": cfg.guardrail.model_dump() if cfg.guardrail else None,
        "positive_class": cfg.positive_class,
        "custom_metric": cfg.custom_metric.model_dump() if cfg.custom_metric else None,
    }


# ---- check-data ----


def load_tokenizer(profile):
    """Tokenizer for a profile (module-level hook so tests can inject a stub)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        profile.hf_id, revision=profile.revision, trust_remote_code=profile.trust_remote_code
    )


def _refuse_resplit_after_noise(exp_dir: Path) -> None:
    if any(r["stage"] == "noise" for r in store.list_requests(exp_dir)):
        raise ValueError("splits are baselined; start a new experiment to resplit the data")


def _refuse_bad_splits(report: dict, task: str, labels: list[str], val: list[dict], test: list[dict]) -> None:
    """Errors and unseen val/test labels would crash training or promote later: refuse before freezing."""
    problems = list(report["errors"])
    if report["unseen_labels"]:
        problems.append(f"labels in val/test but not train: {report['unseen_labels']}")
    if task == "token-cls":
        tags = sorted({t for rows in (val, test) for r in rows for t in r["tags"]} - set(labels))
        if tags:
            problems.append(f"tags in val/test but not train: {tags}")
    if problems:
        raise ValueError("data check failed (nothing was frozen): " + "; ".join(problems))


def _carve_reason(origin: str, hf: bool) -> str | None:
    prefix = "carved from train"
    if not origin.startswith(prefix):
        return None
    rest = origin[len(prefix):].strip()
    if rest.startswith("(") and rest.endswith(")"):
        rest = rest[1:-1]
    return rest or ("no published split" if hf else "single file given")


def check_data(exp_dir: Path, *, resplit: bool = False, val_frac: float = 0.1) -> dict:
    exp_dir = Path(exp_dir)
    root = store.project_root(exp_dir)
    cfg, _, _ = ex.load(exp_dir)
    ex.require_setup(cfg)
    splits = exp_dir / "splits"
    if not 0 < val_frac < 0.9:
        raise ValueError("--val-frac must be between 0 and 0.9")
    frozen_msg = "splits already frozen; --resplit discards them and resets the experiment state"
    if splits.exists() and not resplit:
        raise ValueError(frozen_msg)
    if resplit:
        _refuse_resplit_after_noise(exp_dir)
    # Everything that can fail (data errors, tokenizer download) runs before anything is written.
    task = cfg.task
    got = sources.resolve_source(cfg.data, root, task, val_frac)
    train, val, test, split_origin = got.train, got.val, got.test, got.origin
    if task == "seq-cls":
        data.check_label_column(train + val + test)
    profile = store.profile_for(cfg, {}, root)
    report = data.check(
        exp_dir, task, profile.max_len, load_tokenizer(profile),
        {"train": train, "val": val, "test": test},
    )
    labels = data.label_map(train, task)
    _refuse_bad_splits(report, task, labels, val, test)
    positive = cfg.positive_class or (labels[1] if task == "seq-cls" and len(labels) > 1 else None)
    try:
        params = {**metrics.get(cfg.objective.metric).params, **cfg.objective.params}
    except KeyError:  # custom metric
        params = dict(cfg.objective.params)
    warning = data.val_size_warning(
        task, cfg.objective.metric, params, report["label_dist"].get("val", {}), positive
    )
    if warning:
        report["warnings"].append(warning)
    report["split_origin"] = split_origin
    report["ignored_splits"] = got.ignored_splits
    for role in ("val", "test"):
        reason = _carve_reason(split_origin[role], bool(cfg.data.source))
        if reason:
            report["warnings"].append(f"{role} carved from train: {reason}")

    with ex.state_lock(exp_dir):
        if splits.exists():
            if not resplit:
                raise ValueError(frozen_msg)
            _refuse_resplit_after_noise(exp_dir)
            shutil.rmtree(splits)
            (vault_dir(cfg.uid) / "test.jsonl").unlink(missing_ok=True)
            ex.save_state(exp_dir, ex.State())
        data.freeze_splits(exp_dir, cfg.uid, train, val, test)
        (exp_dir / "data-report.json").write_text(json.dumps(report, indent=2))
        ex.record_checksums(exp_dir, root)
    return report


# ---- compare ----


def _num(v: str):
    if v == "":
        return None
    try:
        f = float(v)
    except ValueError:
        return v
    if math.isnan(f):
        return None
    return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f


def result_rows(exp_dir: Path) -> list[dict]:
    p = Path(exp_dir) / "results.tsv"
    if not p.is_file():
        return []
    with p.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = []
    for r in rows:
        row = {k: (_num(v) if k not in ("description", "config", "verdict", "stage", "run_id") else v)
               for k, v in r.items()}
        row["config"] = json.loads(r["config"]) if r.get("config") else {}
        out.append(row)
    return out


def compare(exp_dir: Path) -> dict:
    exp_dir = Path(exp_dir)
    cfg, _, state = ex.load(exp_dir)
    ex.require_setup(cfg)
    _, metric_delta = store.metric_info(cfg, cfg.objective)
    min_delta = cfg.min_delta if cfg.min_delta is not None else metric_delta
    sig = state.sigma_obj
    n = cfg.seeds
    bar = max(2 * sig * math.sqrt(2 / max(n, 1)), min_delta) if sig is not None else None
    budget = {
        "runs": cfg.budget.max_runs - state.runs_used,
        "hours": round(cfg.budget.max_hours - state.hours_used, 3),
    }
    return {
        "best": state.best_run,
        "noise": {
            "sigma_obj": sig, "sigma_guard": state.sigma_guard, "noise_run": state.noise_run,
            "keep_bar": bar,
        },
        "streak": state.non_keep_streak,
        "stop": state.non_keep_streak >= STREAK_STOP or budget["runs"] <= 0 or budget["hours"] <= 0,
        "budget_left": budget,
        "runs": result_rows(exp_dir),
    }


# ---- misses ----


def _best_seed(exp_dir: Path, cfg: ex.ExperimentConfig, run_id: str) -> int:
    direction, _ = store.metric_info(cfg, cfg.objective)
    ms = [m for m in store.seed_metrics(exp_dir, run_id) if m.get("objective") is not None]
    if not ms:
        raise ValueError(f"run {run_id} has no finished seeds")
    pick = max if direction == "max" else min
    return pick(ms, key=lambda m: m["objective"])["seed"]


def _trim(text: str) -> str:
    return text[:TEXT_CHARS]


def misses(exp_dir: Path, run_id: str, k: int = 20) -> dict:
    import numpy as np

    exp_dir = Path(exp_dir)
    root = store.project_root(exp_dir)
    cfg, _, _ = ex.load(exp_dir)
    ex.require_setup(cfg)
    seed = _best_seed(exp_dir, cfg, run_id)
    seed_dir = store.run_dir(exp_dir, run_id) / f"seed_{seed}"
    val = data.load_rows(exp_dir / "splits" / "val.jsonl", cfg.task)  # never the vault
    if cfg.task == "token-cls":
        pred_path = seed_dir / "val_preds.json"
        if not pred_path.is_file():
            raise FileNotFoundError(f"no val predictions for {run_id} seed {seed}")
        return {"seed": seed, "rows": _token_misses(val, json.loads(pred_path.read_text()), k)}

    pred_path = seed_dir / "val_preds.npy"
    if not pred_path.is_file():
        raise FileNotFoundError(f"no val predictions for {run_id} seed {seed}")
    probs = np.load(pred_path)
    if len(probs) != len(val):
        raise ValueError(f"{len(probs)} predictions but {len(val)} val rows")
    train = data.load_rows(exp_dir / "splits" / "train.jsonl", cfg.task)
    labels = data.label_map(train, cfg.task)
    pos = 1 if len(labels) > 1 else 0
    if cfg.positive_class is not None:
        pos = labels.index(cfg.positive_class)
    idx = {l: i for i, l in enumerate(labels)}
    truth = np.array([idx[str(r["label"])] for r in val])

    thr = None
    for ref in (cfg.objective, cfg.guardrail):
        if ref is None:
            continue
        m = _metric(cfg, ref, root)
        if m.constrained:
            thr = metrics.threshold_for(m, truth, probs, ref.params, pos)
            break
    score = probs[:, pos]
    pred_pos = score >= thr if thr is not None else probs.argmax(1) == pos
    is_pos = truth == pos

    def item(i: int, s: float) -> dict:
        return {"id": val[i]["id"], "text": _trim(val[i]["text"]), "label": str(val[i]["label"]), "score": float(s)}

    fn = sorted(np.nonzero(is_pos & ~pred_pos)[0], key=lambda i: score[i])[:k]
    fp = sorted(np.nonzero(~is_pos & pred_pos)[0], key=lambda i: -score[i])[:k]
    wrong = probs.copy()
    wrong[np.arange(len(val)), truth] = 0.0
    conf = wrong.max(1)
    issues = sorted(np.nonzero(conf > LABEL_ISSUE_SCORE)[0], key=lambda i: -conf[i])[:k]
    return {
        "seed": seed,
        "threshold": thr,
        "false_negatives": [item(i, score[i]) for i in fn],
        "false_positives": [item(i, score[i]) for i in fp],
        "possible_label_issues": [
            {**item(i, conf[i]), "predicted": labels[int(wrong[i].argmax())]} for i in issues
        ],
    }


def _spans(tags: list[str], tokens: list[str]) -> list[tuple]:
    from seqeval.metrics.sequence_labeling import get_entities

    return [(t, s, e + 1, " ".join(tokens[s : e + 1])) for t, s, e in get_entities(tags)]


def _token_misses(val: list[dict], preds: list[list[str]], k: int) -> list[dict]:
    if len(preds) != len(val):
        raise ValueError(f"{len(preds)} predictions but {len(val)} val rows")
    bad = []
    for row, pred in zip(val, preds):
        gold, got = _spans(row["tags"], row["tokens"]), _spans(pred, row["tokens"])
        n_err = len(set(gold) ^ set(got))
        if n_err:
            bad.append((n_err, {"id": row["id"], "text": _trim(" ".join(row["tokens"])), "gold": gold, "pred": got}))
    bad.sort(key=lambda x: -x[0])
    return [b for _, b in bad[:k]]


# ---- next ----
_MORE_ROUND = {"stage": "hparams", "reason": "open more settings for another research round"}


def _research_finished(exp_dir: Path, n: int) -> bool:
    log = exp_dir / "log.md"
    if not log.is_file():
        return False
    for line in log.read_text(errors="replace").splitlines():
        line = line.rstrip()
        if line.startswith(f"## Summary (round {n})") or (n <= 1 and line == "## Summary"):
            return True
    return False


def next_stage(root: Path, exp_dir: Path | None) -> dict:
    """Which stage comes next, from the experiment's files and state. First unmet condition wins."""
    if exp_dir is None:
        exp_root = Path(root) / "experiments"
        numbered = [(int(m.group(1)), d) for d in exp_root.iterdir()
                    if d.is_dir() and (m := re.match(r"(\d+)-", d.name))] if exp_root.is_dir() else []
        if not numbered:
            return {"experiment": None, "stage": "setup", "reason": "no experiment yet", "alternatives": []}
        exp_dir = max(numbered, key=lambda nd: nd[0])[1]
    exp_dir = Path(exp_dir)
    cfg, space, state = ex.load(exp_dir)

    def out(stage: str, reason: str, alternatives: list[dict] | None = None) -> dict:
        return {"experiment": exp_dir.name, "stage": stage, "reason": reason, "alternatives": alternatives or []}

    if ex.missing_setup(cfg) or not (exp_dir / "splits" / "train.jsonl").is_file():
        return out("check", "data not checked and split yet")
    if cfg.bakeoff and not state.model_locked:
        return out("bakeoff", "a bake-off was requested and no model is chosen yet")
    if not (exp_dir / ex.SPACE_FILE).is_file():
        return out("hparams", "no search space committed yet")
    if state.sigma_obj is None:
        return out("baseline", "noise floor not measured yet")
    if not _research_finished(exp_dir, state.round):
        return out("research", f"research round {state.round} is not finished")
    report, log = exp_dir / "report.md", exp_dir / "log.md"
    if not report.is_file() or report.stat().st_mtime < log.stat().st_mtime:
        return out("analyze", "results not analyzed yet", [dict(_MORE_ROUND)])
    if state.promote_pending:
        return out("promote", "promote is running")
    if not state.promoted:
        return out("promote", "final model not trained and scored yet", [dict(_MORE_ROUND)])
    return out("setup", "this experiment is complete; start a new one")
