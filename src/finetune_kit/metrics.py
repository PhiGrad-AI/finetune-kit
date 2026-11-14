"""Metrics registry for seq-cls (probabilities) and token-cls (tag lists)."""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from seqeval.metrics import accuracy_score as _seq_accuracy
from seqeval.metrics import classification_report as _seq_report
from seqeval.metrics import f1_score as _seq_f1
from sklearn import metrics as skm

SEQ = "seq-cls"
TOK = "token-cls"


@dataclass(frozen=True)
class Metric:
    name: str
    tasks: tuple[str, ...]
    direction: Literal["max", "min"]
    min_delta: float
    params: dict = field(default_factory=dict)
    needs_positive: bool = False
    constrained: Literal["recall", "precision"] | None = None
    fn: Callable = None  # type: ignore[assignment]


# ---- recall-constrained helpers (ported from gate_autoresearch/eval.py) ----


def _threshold(pos_scores: np.ndarray, recall: float) -> float:
    """Largest t with share of positives scoring >= t at least `recall`."""
    k = math.ceil(recall * len(pos_scores) - 1e-9)
    return float(np.sort(pos_scores)[::-1][max(k, 1) - 1])


def _binary(y, probs, pos):
    y = np.asarray(y)
    probs = np.asarray(probs, dtype=float)
    return (y == pos).astype(int), probs[:, pos]


def _pauc_hi(y, probs, params, pos):
    yb, s = _binary(y, probs, pos)
    p, n = s[yb == 1], s[yb == 0]
    skips = [
        float(np.mean(n < _threshold(p, r)))
        for r in np.linspace(params["recall_min"], 1.0, int(params["grid"]))
    ]
    return float(np.mean(skips))


def _skip_at_recall(y, probs, params, pos):
    yb, s = _binary(y, probs, pos)
    return float(np.mean(s < _threshold(s[yb == 1], params["recall"])))


def _precision_at_recall(y, probs, params, pos):
    yb, s = _binary(y, probs, pos)
    sel = s >= _threshold(s[yb == 1], params["recall"])
    return float(yb[sel].mean()) if sel.any() else 0.0


def _precision_curve(yb, s):
    """Yield (t, precision, recall) for every distinct score threshold, ascending t."""
    order = np.argsort(-s, kind="stable")
    ys, ss = yb[order], s[order]
    tp = np.cumsum(ys)
    last = np.r_[ss[1:] != ss[:-1], True]  # last index of each distinct score
    idx = np.nonzero(last)[0]
    prec = tp[idx] / (idx + 1)
    rec = tp[idx] / max(int(yb.sum()), 1)
    return ss[idx][::-1], prec[::-1], rec[::-1]


def _recall_at_precision(y, probs, params, pos):
    yb, s = _binary(y, probs, pos)
    _, prec, rec = _precision_curve(yb, s)
    ok = prec >= params["precision"]
    return float(rec[ok].max()) if ok.any() else 0.0


# ---- standard seq-cls metrics ----


def _pred(probs):
    return np.asarray(probs).argmax(1)


def _accuracy(y, probs, params, pos):
    return float(skm.accuracy_score(y, _pred(probs)))


def _f1(y, probs, params, pos):
    probs = np.asarray(probs)
    if probs.shape[1] != 2:
        raise ValueError("f1 requires a binary task; use macro_f1 or micro_f1")
    return float(skm.f1_score(np.asarray(y) == pos, _pred(probs) == pos, zero_division=0))


def _macro_f1(y, probs, params, pos):
    return float(skm.f1_score(y, _pred(probs), average="macro", zero_division=0))


def _micro_f1(y, probs, params, pos):
    return float(skm.f1_score(y, _pred(probs), average="micro", zero_division=0))


def _roc_auc(y, probs, params, pos):
    probs = np.asarray(probs, dtype=float)
    if probs.shape[1] == 2:
        return float(skm.roc_auc_score(np.asarray(y) == pos, probs[:, pos]))
    return float(
        skm.roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=list(range(probs.shape[1])))
    )


def _pr_auc(y, probs, params, pos):
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y)
    if probs.shape[1] == 2:
        return float(skm.average_precision_score(y == pos, probs[:, pos]))
    return float(np.mean([skm.average_precision_score(y == c, probs[:, c]) for c in range(probs.shape[1])]))


def _log_loss(y, probs, params, pos):
    probs = np.asarray(probs, dtype=float)
    return float(skm.log_loss(y, probs, labels=list(range(probs.shape[1]))))


# ---- token-cls metrics ----


def _entity_f1(true_tags, pred_tags, params):
    t = params.get("type")
    if not t:
        return float(_seq_f1(true_tags, pred_tags, zero_division=0))
    rep = _seq_report(true_tags, pred_tags, output_dict=True, zero_division=0)
    return float(rep.get(t, {}).get("f1-score", 0.0))


def _token_accuracy(true_tags, pred_tags, params):
    return float(_seq_accuracy(true_tags, pred_tags))


def _m(name, task, fn, direction="max", min_delta=0.002, params=None, needs_positive=False, constrained=None):
    return Metric(name, (task,), direction, min_delta, params or {}, needs_positive, constrained, fn)


_REGISTRY: dict[str, Metric] = {
    m.name: m
    for m in [
        _m("accuracy", SEQ, _accuracy),
        _m("f1", SEQ, _f1, needs_positive=True),
        _m("macro_f1", SEQ, _macro_f1),
        _m("micro_f1", SEQ, _micro_f1),
        _m("roc_auc", SEQ, _roc_auc),
        _m("pr_auc", SEQ, _pr_auc, needs_positive=True),
        _m("log_loss", SEQ, _log_loss, direction="min", min_delta=0.005),
        _m("pauc_hi", SEQ, _pauc_hi, params={"recall_min": 0.98, "grid": 2001}, needs_positive=True),
        _m("skip_at_recall", SEQ, _skip_at_recall, params={"recall": 0.995}, needs_positive=True, constrained="recall"),
        _m(
            "precision_at_recall",
            SEQ,
            _precision_at_recall,
            params={"recall": 0.995},
            needs_positive=True,
            constrained="recall",
        ),
        _m(
            "recall_at_precision",
            SEQ,
            _recall_at_precision,
            params={"precision": 0.95},
            needs_positive=True,
            constrained="precision",
        ),
        _m("entity_f1", TOK, _entity_f1),
        _m("token_accuracy", TOK, _token_accuracy),
    ]
}


def get(name: str) -> Metric:
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name.startswith("entity_f1_") and len(name) > len("entity_f1_"):
        return replace(_REGISTRY["entity_f1"], name=name, params={"type": name[len("entity_f1_") :]})
    raise KeyError(f"unknown metric: {name}")


def list_for(task: str) -> list[Metric]:
    return [m for m in _REGISTRY.values() if task in m.tasks]


def compute(name: str, params: dict, task: str, y_or_tags, probs_or_pred, pos: int = 1) -> float:
    m = get(name)
    if task not in m.tasks:
        raise ValueError(f"metric {name} does not support task {task}")
    merged = {**m.params, **(params or {})}
    if task == TOK:
        return m.fn(y_or_tags, probs_or_pred, merged)
    return m.fn(y_or_tags, probs_or_pred, merged, pos)


def threshold_for(metric: Metric, y, probs, params: dict, pos: int) -> float | None:
    """Score threshold implied by the metric's constraint, or None."""
    merged = {**metric.params, **(params or {})}
    yb, s = _binary(y, probs, pos)
    if metric.constrained == "recall":
        return _threshold(s[yb == 1], merged["recall"])
    if metric.constrained == "precision":
        ts, prec, _ = _precision_curve(yb, s)
        ok = prec >= merged["precision"]
        return float(ts[ok][0]) if ok.any() else None
    return None


def load_custom(path: Path, func: str, direction: Literal["max", "min"] = "max") -> Metric:
    """Load a seq-cls metric `func(y, probs) -> float` from a Python file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"custom metric file not found: {path}")
    spec = importlib.util.spec_from_file_location(f"_custom_metric_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    user_fn = getattr(mod, func, None)
    if user_fn is None:
        raise ValueError(f"function {func!r} not found in {path}")

    def fn(y, probs, params, pos):
        return float(user_fn(y, probs))

    return Metric(func, (SEQ,), direction, 0.002, {}, False, None, fn)
