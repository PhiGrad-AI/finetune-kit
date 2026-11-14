import json
import random

import numpy as np
import pytest

from finetune_kit import metrics
from finetune_kit.data import freeze_splits, label_map
from finetune_kit.experiment import DataPaths, ExperimentConfig, MetricRef, new_experiment
from finetune_kit.profiles import load_profiles
from finetune_kit.train import worker
from finetune_kit.train.config import build_config
from finetune_kit.train.loop import epoch_order, train_and_predict

pytestmark = [
    pytest.mark.model,
    pytest.mark.filterwarnings("ignore:.*is_gradient_checkpointing.*:FutureWarning"),  # peft
]

WORDS = ["alpha", "beta", "gamma", "delta", "omega", "sigma"]


@pytest.fixture(autouse=True)
def _profiles(monkeypatch):
    monkeypatch.setenv("FINETUNE_KIT_TEST_PROFILES", "1")
    monkeypatch.setenv("FINETUNE_KIT_DEVICE", "cpu")


@pytest.fixture
def prof():
    return load_profiles(None)["tiny-bert"]


def _seq(n, n_classes, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        label = i % n_classes
        text = " ".join(rng.choices(WORDS, k=6)) + (" pii" * label)
        rows.append({"id": f"s{seed}-{i}", "text": text, "label": str(label)})
    return rows


def _tok(n, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        toks = rng.choices(WORDS, k=rng.randint(3, 7))
        tags = ["B-PER" if t == "alpha" else "O" for t in toks]
        rows.append({"id": f"t{seed}-{i}", "tokens": toks, "tags": tags})
    return rows


@pytest.fixture
def seq_rows():
    return _seq(40, 2, 0), _seq(16, 2, 1)


@pytest.fixture
def seq3_rows():
    return _seq(45, 3, 0), _seq(18, 3, 1)


@pytest.fixture
def tok_rows():
    return _tok(40, 0), _tok(16, 1)


def _run(prof, rows, task="seq-cls", **kw):
    train, val = rows
    labels = label_map(train, task)
    cfg = build_config({"max_steps": 4, **kw}, prof, task, 0)
    if task == "seq-cls":
        y = [labels.index(str(r["label"])) for r in val]
        select = lambda p: metrics.compute("accuracy", {}, task, y, p)
    else:
        select = lambda p: metrics.compute("entity_f1", {}, task, [r["tags"] for r in val], p)
    return train_and_predict(cfg, prof, train, val, labels, select, "cpu")


def test_seq_binary_runs_and_scores(prof, seq_rows):
    res = _run(prof, seq_rows)
    assert res.val_out.shape == (16, 2)
    assert np.allclose(res.val_out.sum(1), 1.0, atol=1e-5)
    assert res.train_s > 0 and res.steps == 4 and res.model is None


def test_seq_multiclass_runs(prof, seq3_rows):
    res = _run(prof, seq3_rows)
    assert res.val_out.shape == (18, 3)
    assert np.allclose(res.val_out.sum(1), 1.0, atol=1e-5)


@pytest.mark.parametrize("crf", [True, False])
def test_token_runs(prof, tok_rows, crf):
    res = _run(prof, tok_rows, task="token-cls", crf=crf)
    assert [len(t) for t in res.val_out] == [len(r["tokens"]) for r in tok_rows[1]]
    assert all(t in {"O", "B-PER"} for tags in res.val_out for t in tags)


def test_token_truncated_words_are_O(prof, tok_rows):
    train, val = tok_rows
    res = _run(prof, (train, val), task="token-cls", crf=True, max_len=32)
    assert [len(t) for t in res.val_out] == [len(r["tokens"]) for r in val]
    cfg = build_config({"max_steps": 2, "max_len": 5}, prof, "token-cls", 0)
    labels = label_map(train, "token-cls")
    out = train_and_predict(cfg, prof, train, val, labels, lambda p: 0.0, "cpu")
    assert out.n_truncated > 0
    for row, tags in zip(val, out.val_out):
        assert len(tags) == len(row["tokens"]) and tags[4:] == ["O"] * (len(tags) - 4)


@pytest.mark.parametrize(
    "flags",
    [{"ema": 0.9}, {"swa": True}, {"fgm_eps": 0.5}, {"rdrop_alpha": 1.0}, {"precision": "bf16"}],
)
def test_ema_swa_fgm_rdrop_bf16_smoke(prof, seq_rows, flags):
    res = _run(prof, seq_rows, **flags)
    assert np.isfinite(res.val_out).all()


def test_rdrop_token_cls_smoke(prof, tok_rows):
    res = _run(prof, tok_rows, task="token-cls", rdrop_alpha=1.0, fgm_eps=0.3)
    assert len(res.val_out) == 16


def test_ema_and_swa_together_rejected(prof, seq_rows):
    with pytest.raises(ValueError, match="ema and swa"):
        _run(prof, seq_rows, ema=0.9, swa=True)


@pytest.mark.parametrize(
    "kw",
    [
        {"pooling": "weighted_layers"},
        {"msd_samples": 4},
        {"hidden_dropout": 0.2},
        {"beta2": 0.98},
        {"oversample": 3},
        {"lora_r": 8},
        {"reinit_top": 1, "freeze_embed": True, "class_weight": "balanced"},
    ],
)
def test_new_tunables_smoke(prof, seq_rows, kw):
    res = _run(prof, seq_rows, **kw)
    assert np.isfinite(res.val_out).all()


def test_oversample_epoch_composition():
    labels = [1] * 10 + [0] * 90
    order = epoch_order(labels, 3, random.Random(0))
    assert sum(labels[i] == 1 for i in order) == 30
    assert sum(labels[i] == 0 for i in order) == 90
    assert len(epoch_order(labels, 1, random.Random(0))) == 100


def test_oversample_changes_step_count(prof):
    train = _seq(40, 2, 0)
    cfg = build_config({"oversample": 3, "epochs": 1}, prof, "seq-cls", 0)
    labels = label_map(train, "seq-cls")
    res = train_and_predict(cfg, prof, train, _seq(8, 2, 1), labels, lambda p: 0.0, "cpu")
    assert res.steps == round((40 + 2 * 20) / 8)


def test_same_seed_same_scores(prof, seq_rows):
    a = _run(prof, seq_rows)
    b = _run(prof, seq_rows)
    assert np.allclose(a.val_out, b.val_out, atol=1e-5)


def test_keep_model_returns_model(prof, seq_rows):
    train, val = seq_rows
    cfg = build_config({"max_steps": 2}, prof, "seq-cls", 0)
    res = train_and_predict(
        cfg, prof, train, val, ["0", "1"], lambda p: 0.0, "cpu", keep_model=True
    )
    assert res.model is not None


def test_non_finite_loss_raises(prof, seq_rows, monkeypatch):
    import finetune_kit.train.loop as loop

    monkeypatch.setattr(loop, "seq_loss", lambda *a, **k: a[0].sum() * float("nan"))
    with pytest.raises(RuntimeError, match="non-finite loss at step 0"):
        _run(prof, seq_rows)


@pytest.fixture
def tmp_exp(tmp_path):
    train, val = _seq(40, 2, 0), _seq(16, 2, 1)
    cfg = ExperimentConfig(
        name="t", task="seq-cls", model="tiny-bert", data=DataPaths(train="x"),
        objective=MetricRef(metric="accuracy"), guardrail=MetricRef(metric="roc_auc"),
        positive_class="1",
    )
    d = new_experiment(tmp_path, "t", cfg)
    freeze_splits(d, cfg.uid, train, val, _seq(8, 2, 2))
    return d


def _request(exp, run="r1", **extra):
    rd = exp / "runs" / run
    rd.mkdir(parents=True)
    req = {"stage": "promote", "overrides": {"max_steps": 4}, "seeds": [0, 1],
           "description": "d", "status": "pending", **extra}
    (rd / "request.json").write_text(json.dumps(req))
    return rd


def test_worker_writes_metrics(tmp_exp):
    rd = _request(tmp_exp)
    assert worker.main(["--exp", str(tmp_exp), "--run", "r1", "--seed", "1"]) == 0
    m = json.loads((rd / "seed_1" / "metrics.json").read_text())
    assert {"objective", "guardrail", "report", "train_s", "peak_gb", "n_truncated",
            "best_epoch", "device"} <= set(m)
    assert 0 <= m["objective"] <= 1 and m["guardrail"] is not None
    assert "accuracy" in m["report"] and "f1" in m["report"]
    assert np.load(rd / "seed_1" / "val_preds.npy").shape == (16, 2)


def test_worker_bakeoff_model_override_and_subset(tmp_exp, monkeypatch):
    import yaml

    p = tmp_exp / "experiment.yaml"
    d = yaml.safe_load(p.read_text())
    d["research_subset"] = 20
    p.write_text(yaml.safe_dump(d))
    rd = _request(tmp_exp, stage="bakeoff", overrides={"model": "tiny-roberta", "max_steps": 2})
    assert worker.main(["--exp", str(tmp_exp), "--run", "r1", "--seed", "0"]) == 0
    assert (rd / "seed_0" / "metrics.json").is_file()


def test_worker_token_cls(tmp_path):
    train, val = _tok(40, 0), _tok(16, 1)
    cfg = ExperimentConfig(
        name="tk", task="token-cls", model="tiny-bert", data=DataPaths(train="x"),
        objective=MetricRef(metric="entity_f1"),
    )
    d = new_experiment(tmp_path, "tk", cfg)
    freeze_splits(d, cfg.uid, train, val, _tok(8, 2))
    rd = _request(d, overrides={"max_steps": 3, "crf": True})
    assert worker.main(["--exp", str(d), "--run", "r1", "--seed", "0"]) == 0
    assert len(json.loads((rd / "seed_0" / "val_preds.json").read_text())) == 16
    m = json.loads((rd / "seed_0" / "metrics.json").read_text())
    assert m["guardrail"] is None and "token_accuracy" in m["report"]


def test_worker_error_writes_error_txt(tmp_exp):
    rd = _request(tmp_exp, overrides={"not_a_key": 1})
    assert worker.main(["--exp", str(tmp_exp), "--run", "r1", "--seed", "0"]) == 1
    assert "not_a_key" in (rd / "seed_0" / "error.txt").read_text()
    assert not (rd / "seed_0" / "metrics.json").exists()


def test_worker_fake_train(tmp_exp, monkeypatch):
    monkeypatch.setenv("FINETUNE_KIT_FAKE_TRAIN", "1")
    rd = _request(tmp_exp, seeds=[3, 4], _fake=[0.5, 0.7], _fake_guard=[0.1, 0.2], _fake_train_s=9.0)
    assert worker.main(["--exp", str(tmp_exp), "--run", "r1", "--seed", "4"]) == 0
    m = json.loads((rd / "seed_4" / "metrics.json").read_text())
    assert m["objective"] == 0.7 and m["guardrail"] == 0.2 and m["train_s"] == 9.0
    assert m["report"] == {}
    assert not (rd / "seed_4" / "val_preds.npy").exists()


@pytest.mark.filterwarnings("ignore::UserWarning", "ignore::sklearn.exceptions.UndefinedMetricWarning")
def test_worker_records_metric_value_errors(tmp_exp):
    import yaml

    p = tmp_exp / "experiment.yaml"
    d = yaml.safe_load(p.read_text())
    d["guardrail"] = None  # a failing guardrail is fatal; this test is about report metrics
    p.write_text(yaml.safe_dump(d))
    # val without positives: recall-constrained metrics cannot be computed -> recorded, not dropped
    rows = [r for r in _seq(40, 2, 1) if r["label"] == "0"][:8]
    freeze_splits(tmp_exp, "uid-x", _seq(40, 2, 0), rows, _seq(8, 2, 2))
    rd = _request(tmp_exp)
    assert worker.main(["--exp", str(tmp_exp), "--run", "r1", "--seed", "0"]) == 0
    m = json.loads((rd / "seed_0" / "metrics.json").read_text())
    assert m["report"]["pauc_hi"] is None and "pauc_hi" in m["report_errors"]
    assert "accuracy" in m["report"] and m["report"]["accuracy"] is not None


def test_worker_unexpected_metric_error_fails_without_metrics(tmp_exp, monkeypatch):
    import dataclasses

    bad = dataclasses.replace(
        metrics.get("log_loss"), fn=lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    real = metrics.list_for
    monkeypatch.setattr(metrics, "list_for", lambda t: [*real(t), bad])
    rd = _request(tmp_exp)
    assert worker.main(["--exp", str(tmp_exp), "--run", "r1", "--seed", "0"]) == 1
    assert "boom" in (rd / "seed_0" / "error.txt").read_text()
    assert not (rd / "seed_0" / "metrics.json").exists()


def test_worker_metrics_written_atomically_after_preds(tmp_exp, monkeypatch):
    seen = {}
    real = worker.os.replace

    def spy(src, dst):
        seen["preds_before"] = (dst.parent / "val_preds.npy").is_file()
        seen["src"] = str(src)
        real(src, dst)

    monkeypatch.setattr(worker.os, "replace", spy)
    rd = _request(tmp_exp)
    assert worker.main(["--exp", str(tmp_exp), "--run", "r1", "--seed", "0"]) == 0
    assert seen["preds_before"] and seen["src"].endswith("metrics.json.tmp")
    assert not list((rd / "seed_0").glob("*.tmp"))


def test_worker_rerun_clears_stale_files(tmp_exp):
    rd = _request(tmp_exp, overrides={"not_a_key": 1})
    sd = rd / "seed_0"
    assert worker.main(["--exp", str(tmp_exp), "--run", "r1", "--seed", "0"]) == 1
    req = json.loads((rd / "request.json").read_text())
    req["overrides"] = {"max_steps": 2}
    (rd / "request.json").write_text(json.dumps(req))
    assert worker.main(["--exp", str(tmp_exp), "--run", "r1", "--seed", "0"]) == 0
    assert (sd / "metrics.json").is_file() and not (sd / "error.txt").exists()
