"""Noise-aware keep/discard statistics."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Literal

Direction = Literal["max", "min"]


@dataclass
class RunSummary:
    objective_mean: float
    objective_std: float
    guardrail_mean: float | None
    n_seeds: int
    n_nondefault: int


def sigma(values: list[float]) -> float:
    """Sample standard deviation (ddof=1); 0.0 for fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _gain(cand: float, best: float, direction: Direction) -> float:
    return cand - best if direction == "max" else best - cand


def _noise(sig: float, n: int) -> float:
    return sig * math.sqrt(2 / max(n, 1))


def verdict(
    cand: RunSummary,
    best: RunSummary,
    sigma_obj: float,
    sigma_guard: float | None,
    direction: Direction,
    min_delta: float,
    guard_direction: Direction = "max",
) -> Literal["keep", "keep-simpler", "discard"]:
    n = min(cand.n_seeds, best.n_seeds)
    delta = _gain(cand.objective_mean, best.objective_mean, direction)
    noise = _noise(sigma_obj, n)

    guard_ok = True
    if (
        sigma_guard is not None
        and cand.guardrail_mean is not None
        and best.guardrail_mean is not None
    ):
        drop = -_gain(cand.guardrail_mean, best.guardrail_mean, guard_direction)
        guard_ok = drop <= _noise(sigma_guard, n)

    if guard_ok and delta > max(2 * noise, min_delta):
        return "keep"
    if guard_ok and abs(delta) <= 2 * noise and cand.n_nondefault < best.n_nondefault:
        return "keep-simpler"
    return "discard"


def bakeoff_rank(
    results: dict[str, tuple[list[float], float]],
    direction: Direction,
    min_delta: float,
) -> dict:
    """Rank models by mean objective; pick a winner, preferring speed on ties."""
    if not results:
        raise ValueError("bakeoff_rank needs at least one model")
    rows = []
    for model, (values, train_s) in results.items():
        rows.append(
            {
                "model": model,
                "mean": statistics.fmean(values),
                "std": sigma(values),
                "train_s": train_s,
            }
        )
    sign = 1 if direction == "max" else -1
    rows.sort(key=lambda r: -sign * r["mean"])
    leader = rows[0]
    if len(rows) == 1:
        return {
            "ranking": rows,
            "winner": leader["model"],
            "clear": True,
            "reason": "only one model",
        }

    n = min(len(v) for v, _ in results.values())
    pooled = math.sqrt(statistics.fmean(r["std"] ** 2 for r in rows))
    bar = max(2 * _noise(pooled, n), min_delta)
    gap = sign * (leader["mean"] - rows[1]["mean"])
    if gap > bar:
        return {
            "ranking": rows,
            "winner": leader["model"],
            "clear": True,
            "reason": f"{leader['model']} leads runner-up by {gap:.4f} > bar {bar:.4f}",
        }
    within = [r for r in rows if sign * (leader["mean"] - r["mean"]) <= bar]
    fastest = min(within, key=lambda r: r["train_s"])
    return {
        "ranking": rows,
        "winner": fastest["model"],
        "clear": False,
        "reason": (
            f"{len(within)} models within bar {bar:.4f} of leader; "
            f"picked fastest ({fastest['model']})"
        ),
    }
