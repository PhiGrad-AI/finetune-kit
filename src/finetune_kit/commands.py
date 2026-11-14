"""Orchestration behind the setup, data and comparison CLI commands (cli.py keeps thin wrappers)."""
import csv
import json
import math
import shutil
from pathlib import Path

from finetune_kit import data, experiment as ex, metrics, store
from finetune_kit.data import DataError
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


# ---- new ----


def setup_experiment(
    root: Path, name: str, *, task: str, model: str, train: str, val: str | None, test: str | None,
    metric: str, metric_params: dict, guardrail: str | None, guardrail_params: dict,
    positive_class: str | None, seeds: int, max_runs: int, max_hours: float, subset: int | None,
    custom_metric: str | None, direction: str | None,
) -> Path:
    from finetune_kit.profiles import load_profiles

    root = Path(root)
    if not ex._slug(name):
        raise ValueError(f"experiment name {name!r} has no letters or digits")
    if task not in ("seq-cls", "token-cls"):
        raise ValueError(f"--task must be seq-cls or token-cls, not {task!r}")
    if (val is None) != (test is None):
        raise ValueError("give both --val and --test, or neither (the kit then splits --train)")
    profiles = load_profiles(root)
    if model not in profiles:
        raise ValueError(f"unknown model {model!r}; see `finetune-kit models`")
    if task not in profiles[model].tasks:
        raise ValueError(f"model {model!r} does not support {task}")
    for label, p in (("train", train), ("val", val), ("test", test)):
        if p is not None and not _resolve(root, p).is_file():
            raise FileNotFoundError(f"--{label} file not found: {p}")

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

    if task == "seq-cls":
        labels = data.label_map(data.load_rows(_resolve(root, train), task), task)
        if positive_class is not None and positive_class not in labels:
            raise ValueError(f"--positive-class {positive_class!r} is not a train label {labels}")
        if positive_class is None and len(labels) > 2 and any(m.needs_positive for m in resolved):
            names = [m.name for m in resolved if m.needs_positive]
            raise ValueError(
                f"metric {', '.join(names)} needs --positive-class: train has {len(labels)} classes {labels}"
            )
    elif positive_class is not None:
        raise ValueError("--positive-class applies to seq-cls only")

    profile = profiles[model]
    cfg = ex.ExperimentConfig(
        name=name, task=task, model=model,
        data=ex.DataPaths(train=train, val=val, test=test),
        objective=ex.MetricRef(metric=metric, params=metric_params),
        guardrail=ex.MetricRef(metric=guardrail, params=guardrail_params) if guardrail else None,
        positive_class=positive_class, custom_metric=cm, seeds=seeds,
        budget=ex.Budget(max_runs=max_runs, max_hours=max_hours), research_subset=subset,
        versions=store.current_versions(profile),
    )
    return ex.new_experiment(root, name, cfg)


# ---- check-data ----


def load_tokenizer(profile):
    """Tokenizer for a profile (module-level hook so tests can inject a stub)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        profile.hf_id, revision=profile.revision, trust_remote_code=profile.trust_remote_code
    )


def check_data(exp_dir: Path, *, resplit: bool = False, val_frac: float = 0.1) -> dict:
    exp_dir = Path(exp_dir)
    root = store.project_root(exp_dir)
    cfg, _, _ = ex.load(exp_dir)
    splits = exp_dir / "splits"
    if not 0 < val_frac < 0.9:
        raise ValueError("--val-frac must be between 0 and 0.9")
    with ex.state_lock(exp_dir):
        if splits.exists():
            if not resplit:
                raise ValueError("splits already frozen; --resplit discards them and resets the experiment state")
            shutil.rmtree(splits)
            (vault_dir(cfg.uid) / "test.jsonl").unlink(missing_ok=True)
            ex.save_state(exp_dir, ex.State())
        task = cfg.task
        train = data.load_rows(_resolve(root, cfg.data.train), task)
        if cfg.data.val and cfg.data.test:
            val = data.load_rows(_resolve(root, cfg.data.val), task)
            test = data.load_rows(_resolve(root, cfg.data.test), task)
        else:
            train, val, test = data.make_splits(train, task, frac=(1 - 0.1 - val_frac, val_frac, 0.1))
        data.freeze_splits(exp_dir, cfg.uid, train, val, test)

        profile = store.profile_for(cfg, {}, root)
        report = data.check(
            exp_dir, task, profile.max_len, load_tokenizer(profile),
            {"train": train, "val": val, "test": test},
        )
        labels = data.label_map(train, task)
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
