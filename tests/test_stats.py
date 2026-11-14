import pytest

from finetune_kit.stats import RunSummary, bakeoff_rank, sigma, verdict

B = RunSummary(0.70, 0.01, 0.40, 3, 0)


def S(obj, guard=0.40, nd=1):
    return RunSummary(obj, 0.01, guard, 3, nd)


def test_keep():
    assert verdict(S(0.75), B, 0.03, 0.02, "max", 0.002) == "keep"


def test_below_bar():
    assert verdict(S(0.745), B, 0.03, 0.02, "max", 0.002) == "discard"


def test_min_delta_wins():
    assert verdict(S(0.7015), B, 0.0, None, "max", 0.002) == "discard"


def test_guardrail_drop():
    assert verdict(S(0.80, guard=0.37), B, 0.03, 0.02, "max", 0.002) == "discard"


def test_guardrail_small_drop_ok():
    assert verdict(S(0.80, guard=0.39), B, 0.03, 0.02, "max", 0.002) == "keep"


def test_guardrail_missing_mean_ok():
    assert verdict(S(0.80, guard=None), B, 0.03, 0.02, "max", 0.002) == "keep"


def test_guardrail_min_direction():
    # guard direction "min": an increase is a drop in quality
    assert verdict(S(0.80, guard=0.43), B, 0.03, 0.02, "max", 0.002, guard_direction="min") == "discard"


def test_simpler():
    best = RunSummary(0.75, 0.01, 0.40, 3, 3)
    assert verdict(S(0.74, nd=1), best, 0.03, 0.02, "max", 0.002) == "keep-simpler"


def test_not_simpler_when_not_fewer_keys():
    best = RunSummary(0.75, 0.01, 0.40, 3, 1)
    assert verdict(S(0.74, nd=1), best, 0.03, 0.02, "max", 0.002) == "discard"


def test_simpler_blocked_by_guardrail():
    best = RunSummary(0.75, 0.01, 0.40, 3, 3)
    assert verdict(S(0.74, guard=0.30, nd=1), best, 0.03, 0.02, "max", 0.002) == "discard"


def test_min_direction():
    assert verdict(S(0.30), RunSummary(0.40, .01, None, 3, 0), 0.03, None, "min", 0.005) == "keep"


def test_n_is_min_of_seeds():
    cand = RunSummary(0.75, 0.01, None, 10, 1)
    best = RunSummary(0.70, 0.01, None, 2, 0)
    # N=2: noise = 0.03 -> bar 0.06 > 0.05
    assert verdict(cand, best, 0.03, None, "max", 0.002) == "discard"


def test_sigma():
    assert sigma([0.7241, 0.6682, 0.6698]) == pytest.approx(0.0318, abs=1e-4)


def test_sigma_short():
    assert sigma([]) == 0.0
    assert sigma([0.5]) == 0.0


def test_bakeoff_clear_winner():
    r = bakeoff_rank({"a": ([.80, .81, .79], 2000.0), "b": ([.70, .71, .69], 1000.0)}, "max", 0.002)
    assert r["winner"] == "a" and r["clear"] is True
    assert [x["model"] for x in r["ranking"]] == ["a", "b"]
    assert r["ranking"][0]["mean"] == pytest.approx(0.80)
    assert r["ranking"][0]["train_s"] == 2000.0
    assert isinstance(r["reason"], str)


def test_bakeoff_tie_picks_faster():
    r = bakeoff_rank({"a": ([.72, .66, .67], 2000.0), "b": ([.70, .64, .69], 1000.0)}, "max", 0.002)
    assert r["clear"] is False and r["winner"] == "b"


def test_bakeoff_min_direction():
    r = bakeoff_rank({"a": ([.30, .31, .29], 10.0), "b": ([.50, .51, .49], 5.0)}, "min", 0.002)
    assert r["winner"] == "a" and r["clear"] is True


def test_bakeoff_single_model():
    r = bakeoff_rank({"a": ([.7, .7, .7], 1.0)}, "max", 0.002)
    assert r["winner"] == "a" and r["clear"] is True
