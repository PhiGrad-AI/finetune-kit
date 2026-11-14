"""Per-seed worker: `python -m finetune_kit.train.worker --exp DIR --run RUN_ID --seed K`."""
import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from finetune_kit import metrics
from finetune_kit.data import label_map, load_rows, subset
from finetune_kit.experiment import ExperimentConfig, MetricRef, load
from finetune_kit.profiles import load_profiles
from finetune_kit.train.config import build_config

SUBSET_STAGES = {"research", "noise", "lr-scan", "bakeoff"}


def _num(v):
    return float(v) if v is not None and math.isfinite(v) else None


def _metric(exp: ExperimentConfig, ref: MetricRef, exp_dir: Path) -> metrics.Metric:
    cm = exp.custom_metric
    if cm is not None and ref.metric == cm.func:
        p = Path(cm.path)
        for cand in (p, exp_dir.parent.parent / p, exp_dir / p):
            if cand.is_file():
                return metrics.load_custom(cand, cm.func, cm.direction)
        raise FileNotFoundError(f"custom metric file not found: {cm.path}")
    return metrics.get(ref.metric)


def _score(m: metrics.Metric, params: dict, task: str, truth, out, pos: int) -> float:
    merged = {**m.params, **(params or {})}
    return m.fn(truth, out, merged) if task == "token-cls" else m.fn(truth, out, merged, pos)


def _write_metrics(seed_dir: Path, out: dict) -> None:
    tmp = seed_dir / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=2))
    os.replace(tmp, seed_dir / "metrics.json")


def _fake(request: dict, seed: int, seed_dir: Path) -> None:
    seeds = request["seeds"]
    i = seeds.index(seed)
    if request.get("_fake_sleep"):
        time.sleep(float(request["_fake_sleep"]))
    guard = request.get("_fake_guard", [None] * len(seeds))[i]
    out = {
        "objective": request["_fake"][i], "guardrail": guard, "report": {},
        "train_s": request.get("_fake_train_s", 1.0), "peak_gb": 0.0, "n_truncated": 0,
        "best_epoch": 0.0, "device": "fake",
    }
    _write_metrics(seed_dir, out)


def _real(exp_dir: Path, request: dict, seed: int, seed_dir: Path) -> None:
    from finetune_kit.train.loop import pick_device, train_and_predict

    exp, _, _ = load(exp_dir)
    task = exp.task
    overrides = dict(request.get("overrides") or {})
    model_key = overrides.pop("model", exp.model)
    profiles = load_profiles(exp_dir.parent.parent)
    if model_key not in profiles:
        raise ValueError(f"unknown model profile {model_key!r}")
    profile = profiles[model_key]
    cfg = build_config(overrides, profile, task, seed)

    train_rows = load_rows(exp_dir / "splits" / "train.jsonl", task)
    val_rows = load_rows(exp_dir / "splits" / "val.jsonl", task)
    if exp.research_subset and request.get("stage") in SUBSET_STAGES:
        train_rows = subset(train_rows, exp.research_subset, 0)
    labels = label_map(train_rows, task)

    pos = 1 if len(labels) > 1 else 0
    if exp.positive_class is not None:
        if exp.positive_class not in labels:
            raise ValueError(f"positive_class {exp.positive_class!r} not in labels {labels}")
        pos = labels.index(exp.positive_class)

    if task == "seq-cls":
        idx = {l: i for i, l in enumerate(labels)}
        unseen = sorted({str(r["label"]) for r in val_rows} - set(idx))
        if unseen:
            raise ValueError(f"val labels not present in train: {unseen}")
        truth = [idx[str(r["label"])] for r in val_rows]
    else:
        truth = [r["tags"] for r in val_rows]

    obj_m = _metric(exp, exp.objective, exp_dir)
    guard_m = _metric(exp, exp.guardrail, exp_dir) if exp.guardrail else None

    def select(out):
        return _score(obj_m, exp.objective.params, task, truth, out, pos)

    device = pick_device()
    res = train_and_predict(
        cfg, profile, train_rows, val_rows, labels, select, device,
        direction=obj_m.direction, pos=pos,
    )
    objective = select(res.val_out)
    guardrail = (
        _score(guard_m, exp.guardrail.params, task, truth, res.val_out, pos) if guard_m else None
    )
    report, report_errors = {}, {}
    for m in metrics.list_for(task):
        if m.needs_positive and len(labels) > 2:
            continue
        try:
            report[m.name] = _num(_score(m, {}, task, truth, res.val_out, pos))
        except (ValueError, IndexError) as e:  # not applicable, e.g. val without positives; others are bugs
            report[m.name] = None
            report_errors[m.name] = str(e)
    out = {
        "objective": _num(objective), "guardrail": _num(guardrail), "report": report,
        "report_errors": report_errors,
        "train_s": res.train_s, "peak_gb": res.peak_gb, "n_truncated": res.n_truncated,
        "best_epoch": res.best_epoch, "device": device,
    }
    # preds first; metrics.json is the completion marker, so it is written last and atomically
    if task == "seq-cls":
        np.save(seed_dir / "val_preds.npy", res.val_out)
    else:
        (seed_dir / "val_preds.json").write_text(json.dumps(res.val_out))
    _write_metrics(seed_dir, out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)
    run_dir = a.exp / "runs" / a.run
    seed_dir = run_dir / f"seed_{a.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("metrics.json", "error.txt"):  # a rerun never leaves both behind
        (seed_dir / stale).unlink(missing_ok=True)
    try:
        request = json.loads((run_dir / "request.json").read_text())
        if os.environ.get("FINETUNE_KIT_FAKE_TRAIN") == "1":
            _fake(request, a.seed, seed_dir)
        else:
            _real(a.exp, request, a.seed, seed_dir)
    except Exception as e:
        (seed_dir / "metrics.json").unlink(missing_ok=True)
        (seed_dir / "error.txt").write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
