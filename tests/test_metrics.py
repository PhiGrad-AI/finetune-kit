import numpy as np
import pytest

from finetune_kit.metrics import compute, get, list_for, load_custom, threshold_for


def P(s):
    s = np.asarray(s)
    return np.stack([1 - s, s], 1)


Y = np.array([0, 0, 1, 1])
S = P([0.1, 0.9, 0.5, 0.95])


def test_pauc_hi_perfect():
    assert compute("pauc_hi", {}, "seq-cls", Y, P([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_pauc_hi_overlap():
    assert compute("pauc_hi", {}, "seq-cls", Y, S) == pytest.approx(0.5)


def test_skip_at_recall():
    assert compute("skip_at_recall", {}, "seq-cls", Y, S) == pytest.approx(0.25)


def test_precision_at_recall():
    assert compute("precision_at_recall", {}, "seq-cls", Y, S) == pytest.approx(2 / 3)


def test_recall_at_precision():
    assert compute("recall_at_precision", {"precision": 1.0}, "seq-cls", Y, S) == pytest.approx(0.5)
    assert compute("recall_at_precision", {"precision": 0.6}, "seq-cls", Y, S) == pytest.approx(1.0)


def test_positive_class_multiclass():
    y = np.array([0, 2, 2, 1])
    probs = np.eye(3)[[0, 2, 2, 1]]
    assert compute("pauc_hi", {}, "seq-cls", y, probs, pos=2) == 1.0


def test_standard_metrics():
    y = np.array([0, 1, 1, 0])
    probs = P([0.2, 0.8, 0.4, 0.6])
    assert compute("accuracy", {}, "seq-cls", y, probs) == pytest.approx(0.5)
    assert compute("f1", {}, "seq-cls", y, probs) == pytest.approx(0.5)
    assert compute("macro_f1", {}, "seq-cls", y, probs) == pytest.approx(0.5)
    assert compute("micro_f1", {}, "seq-cls", y, probs) == pytest.approx(0.5)
    assert compute("roc_auc", {}, "seq-cls", y, probs) == pytest.approx(0.75)
    assert compute("pr_auc", {}, "seq-cls", y, probs) == pytest.approx(5 / 6)
    expected = -np.mean(np.log([0.8, 0.8, 0.4, 0.4]))
    assert compute("log_loss", {}, "seq-cls", y, probs) == pytest.approx(expected)


def test_f1_multiclass_raises():
    with pytest.raises(ValueError, match="binary"):
        compute("f1", {}, "seq-cls", np.array([0, 1, 2]), np.eye(3))


def test_entity_f1():
    assert compute("entity_f1", {}, "token-cls", [["B-PER", "I-PER", "O"]], [["B-PER", "I-PER", "O"]]) == 1.0
    assert compute("entity_f1", {"type": "LOC"}, "token-cls", [["B-LOC", "O"]], [["O", "O"]]) == 0.0
    assert compute("entity_f1_PER", {}, "token-cls", [["B-PER", "O"]], [["B-PER", "O"]]) == 1.0


def test_token_accuracy():
    assert compute("token_accuracy", {}, "token-cls", [["B-PER", "O"]], [["B-PER", "B-PER"]]) == 0.5


def test_directions():
    assert get("log_loss").direction == "min" and get("pauc_hi").direction == "max"
    assert get("log_loss").min_delta == 0.005
    assert get("accuracy").min_delta == 0.002


def test_defaults_and_overrides():
    assert get("pauc_hi").params == {"recall_min": 0.98, "grid": 2001}
    assert get("recall_at_precision").params == {"precision": 0.95}
    # params merge over defaults
    assert compute("skip_at_recall", {"recall": 0.5}, "seq-cls", Y, S) == pytest.approx(0.75)


def test_list_for():
    assert "pauc_hi" in [m.name for m in list_for("seq-cls")]
    assert "entity_f1" in [m.name for m in list_for("token-cls")]
    assert "pauc_hi" not in [m.name for m in list_for("token-cls")]


def test_threshold_recall():
    assert threshold_for(get("skip_at_recall"), Y, S, {}, 1) == pytest.approx(0.5)


def test_threshold_precision():
    m = get("recall_at_precision")
    assert threshold_for(m, Y, S, {"precision": 1.0}, 1) == pytest.approx(0.95)
    assert threshold_for(m, Y, S, {"precision": 0.6}, 1) == pytest.approx(0.5)
    assert threshold_for(m, np.array([0, 1]), P([0.9, 0.1]), {"precision": 0.9}, 1) is None


def test_threshold_none_for_unconstrained():
    assert threshold_for(get("accuracy"), Y, S, {}, 1) is None


def test_unknown_metric():
    with pytest.raises(KeyError, match="unknown metric"):
        get("nope")


def test_custom_metric(tmp_path):
    (tmp_path / "m.py").write_text("def acc(y, probs): return float((probs.argmax(1)==y).mean())")
    m = load_custom(tmp_path / "m.py", "acc")
    assert m.fn(np.array([1]), P([0.9]), {}, 1) == 1.0
    assert m.direction == "max"
    m2 = load_custom(tmp_path / "m.py", "acc", direction="min")
    assert m2.direction == "min"
