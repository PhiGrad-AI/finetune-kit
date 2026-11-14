import json
import os
import random
import shutil
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from finetune_kit import experiment as ex
from finetune_kit import metrics, store
from finetune_kit.cli import app
from finetune_kit.data import freeze_splits
from finetune_kit.paths import vault_dir
from finetune_kit.promote import promote

pytestmark = [
    pytest.mark.model,
    pytest.mark.filterwarnings("ignore:.*is_gradient_checkpointing.*:FutureWarning"),  # peft
]

ENV = {
    "FINETUNE_KIT_TESTING": "1", "FINETUNE_KIT_TEST_PROFILES": "1", "FINETUNE_KIT_DEVICE": "cpu",
    "FINETUNE_KIT_MAX_STEPS": "3", "FINETUNE_KIT_NO_DAEMON": "1",
}
WORDS = ["alpha", "beta", "gamma", "delta", "omega", "sigma"]
SRC = Path(__file__).resolve().parent.parent / "src" / "finetune_kit"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("FINETUNE_KIT_TIMEOUT_S", raising=False)


def _seq(n, seed):
    rng = random.Random(seed)
    return [
        {"id": f"s{seed}-{i}", "text": " ".join(rng.choices(WORDS, k=5)) + " pii" * (i % 2), "label": str(i % 2)}
        for i in range(n)
    ]


def _tok(n, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        toks = rng.choices(WORDS, k=4) + ["Ann", "Lee"]
        rows.append({"id": f"t{seed}-{i}", "tokens": toks, "tags": ["O"] * 4 + ["B-PER", "I-PER"]})
    return rows


def _make_exp(root: Path, task="seq-cls", objective="skip_at_recall", guardrail="roc_auc") -> Path:
    cfg = ex.ExperimentConfig(
        name="p", task=task, model="tiny-bert", data=ex.DataPaths(train="x"),
        objective=ex.MetricRef(metric=objective),
        guardrail=ex.MetricRef(metric=guardrail) if guardrail else None,
        research_subset=20,
    )
    d = ex.new_experiment(root, "p", cfg)
    if task == "seq-cls":
        freeze_splits(d, cfg.uid, _seq(60, 0), _seq(16, 1), _seq(12, 2))
    else:
        freeze_splits(d, cfg.uid, _tok(40, 0), _tok(12, 1), _tok(8, 2))
    mem = root / ".finetune-kit" / "memory"
    mem.mkdir(parents=True)
    (mem / "research-rules.md").write_text("rules\n")
    ex.record_checksums(d, root)
    noise = _finished(d, "noise", {}, "baseline")
    with ex.state_lock(d):
        state = ex.load(d)[2]
        state.noise_run = state.best_run = noise
        state.sigma_obj, state.baseline_train_s = 0.01, 60.0
        ex.save_state(d, state)
    return d


def _finished(d: Path, stage: str, overrides: dict, verdict: str) -> str:
    """A run recorded as finished with `verdict` (promote reads only the verdict and overrides)."""
    rid = store.new_run(d, stage, overrides, [0, 1], f"{stage} run")
    req = store.read_request(d, rid)
    req.update(status="done", verdict=verdict)
    store.write_request(d, rid, req)
    with ex.state_lock(d):
        state = ex.load(d)[2]
        state.finalized[rid] = verdict
        ex.save_state(d, state)
    return rid


def _module_exp(tmp_path_factory, name, **kw):
    root = tmp_path_factory.mktemp(name)
    with pytest.MonkeyPatch.context() as mp:
        for k, v in ENV.items():
            mp.setenv(k, v)
        mp.setenv("FINETUNE_KIT_HOME", str(root / "kit-home"))
        mp.delenv("FINETUNE_KIT_TIMEOUT_S", raising=False)
        overrides, verdict, seeds = kw.pop("overrides"), kw.pop("verdict"), kw.pop("seeds")
        d = _make_exp(root, **kw)
        rid = _finished(d, "research", overrides, verdict)
        result = promote(d, rid, seeds=seeds, sync=True)
    return d, rid, result


@pytest.fixture(scope="module")
def promoted(tmp_path_factory):
    return _module_exp(tmp_path_factory, "recall", overrides={"pooling": "mean"}, verdict="keep", seeds=2)


@pytest.fixture(scope="module")
def promoted_f1(tmp_path_factory):
    return _module_exp(
        tmp_path_factory, "f1", objective="f1", guardrail="accuracy",
        overrides={"lora_r": 4, "head": "mlp", "pooling": "attention"}, verdict="keep-simpler", seeds=1,
    )


@pytest.fixture(scope="module")
def promoted_tok(tmp_path_factory):
    return _module_exp(
        tmp_path_factory, "tok", task="token-cls", objective="entity_f1", guardrail=None,
        overrides={"crf": True, "label_align": "all"}, verdict="keep", seeds=1,
    )


def _val_rows(d):
    return [json.loads(l) for l in (d / "splits" / "val.jsonl").read_text().splitlines()]


def _seed_dir(d, result):
    return store.run_dir(d, result["run_id"]) / f"seed_{result['seed']}"


def _run_predict(d, tmp_path, rows):
    inp, out = tmp_path / "rows.jsonl", tmp_path / "scores.jsonl"
    inp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    p = subprocess.run(
        [sys.executable, str(d / "final" / "predict.py"), "--input", str(inp), "--output", str(out), "--device", "cpu"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert p.returncode == 0, p.stderr
    return [json.loads(l) for l in out.read_text().splitlines()]


def test_promote_writes_final(promoted):
    d, src, result = promoted
    final = d / "final"
    for name in ("recipe.json", "test_metrics.json", "val_metrics.json", "throughput.json", "predict.py"):
        assert (final / name).is_file(), name
    model = final / "model"
    assert (model / "config.json").is_file() and (model / "classifier.pt").is_file()
    assert any(model.glob("tokenizer*")) or (model / "vocab.txt").is_file()
    recipe = json.loads((final / "recipe.json").read_text())
    assert recipe["config"]["pooling"] == "mean" and recipe["config"]["task"] == "seq-cls"
    assert recipe["profile"]["hf_id"] == "hf-internal-testing/tiny-random-bert"
    assert {"finetune-kit", "torch", "transformers"} <= set(recipe["versions"])
    assert recipe["source_run"] == src and recipe["labels"] == ["0", "1"]
    test_m = json.loads((final / "test_metrics.json").read_text())
    assert test_m["n_rows"] == 12 and test_m["promote_run"] == result["run_id"]
    # skip_at_recall on test re-fits its threshold on test labels: labelled, not the headline
    assert "objective" not in test_m and test_m["objective_oracle_threshold"] is not None
    assert test_m["headline"] == {
        "metric": "skip_at_recall", "source": "at_threshold.skip", "value": test_m["at_threshold"]["skip"],
    }
    assert "accuracy" in test_m["report"] and test_m["objective_metric"] == "skip_at_recall"
    state = ex.load(d)[2]
    assert state.promoted is True and state.promote_pending is None
    assert result["status"] == "done" and result["test_metrics"] == test_m
    assert Path(result["final"]).resolve() == final.resolve()
    # the full train split is used (research_subset=20 applies to research runs only)
    rdir = store.run_dir(d, result["run_id"])
    assert json.loads((rdir / "seed_0" / "metrics.json").read_text())["n_train"] == 60
    # no checkpoints kept outside final/
    assert not list(rdir.glob("seed_*/model"))


def test_promote_twice_refused(promoted, runner):
    d, src, _ = promoted
    with pytest.raises(ValueError, match="already promoted; test is scored once"):
        promote(d, src)
    r = runner.invoke(app, ["promote", str(d), src])
    assert r.exit_code == 2 and "already promoted" in json.loads(r.stdout)["error"]


def test_promote_discarded_run_refused(tmp_path):
    d = _make_exp(tmp_path)
    rid = _finished(d, "research", {"dropout": 0.2}, "discard")
    with pytest.raises(ValueError, match="keep"):
        promote(d, rid)
    assert ex.load(d)[2].promote_pending is None


def test_noise_run_accepted_and_pending_guard(tmp_path, runner):
    d = _make_exp(tmp_path)
    noise = ex.load(d)[2].noise_run
    r = runner.invoke(app, ["promote", str(d), noise, "--seeds", "2"])
    assert r.exit_code == 0, r.stdout
    out = json.loads(r.stdout)
    assert out["status"] == "queued"
    req = store.read_request(d, out["run_id"])
    assert req["stage"] == "promote" and req["seeds"] == [0, 1] and req["source_run"] == noise
    state = ex.load(d)[2]
    assert state.promote_pending == out["run_id"] and state.request_hashes[out["run_id"]] == store.request_hash(req)
    r2 = runner.invoke(app, ["promote", str(d), noise])
    assert r2.exit_code == 2 and "pending" in json.loads(r2.stdout)["error"]


def test_threshold_saved_for_recall_metric(promoted):
    d, _, result = promoted
    thr = json.loads((d / "final" / "threshold.json").read_text())
    probs = np.load(_seed_dir(d, result) / "val_preds.npy")
    y = np.array([int(r["label"]) for r in _val_rows(d)])
    expected = metrics.threshold_for(metrics.get("skip_at_recall"), y, probs, {}, 1)
    assert thr["threshold"] == pytest.approx(expected) and thr["metric"] == "skip_at_recall"
    assert result["threshold"] == thr
    at = json.loads((d / "final" / "test_metrics.json").read_text())["at_threshold"]
    assert {"recall", "precision", "skip", "accuracy"} <= set(at)


def test_no_threshold_for_f1(promoted_f1):
    d, _, result = promoted_f1
    assert result["status"] == "done", result
    assert not (d / "final" / "threshold.json").exists() and result["threshold"] is None
    test_m = json.loads((d / "final" / "test_metrics.json").read_text())
    assert "at_threshold" not in test_m
    assert test_m["headline"] == {"metric": "f1", "source": "objective", "value": test_m["objective"]}


def test_throughput_written(promoted):
    d, _, result = promoted
    tp = json.loads((d / "final" / "throughput.json").read_text())
    assert tp["rows_per_s"]["cpu"] > 0 and tp["model_mb"] > 0 and tp["p50_latency_ms"]["cpu"] > 0
    assert result["throughput"] == tp


def test_exported_predict_matches(promoted, tmp_path):
    d, _, result = promoted
    src = (d / "final" / "predict.py").read_text()
    assert "finetune_kit" not in src and "{{" not in src
    rows = [{"id": r["id"], "text": r["text"]} for r in _val_rows(d)[:3]]
    out = _run_predict(d, tmp_path, rows)
    probs = np.load(_seed_dir(d, result) / "val_preds.npy")
    thr = result["threshold"]["threshold"]
    assert [o["id"] for o in out] == [r["id"] for r in rows]
    np.testing.assert_allclose([o["score"] for o in out], probs[:3, 1], atol=1e-5)
    assert [o["flag"] for o in out] == [bool(s >= thr) for s in probs[:3, 1]]


def test_exported_predict_matches_lora_mlp_attention(promoted_f1, tmp_path):
    d, _, result = promoted_f1
    rows = [{"id": r["id"], "text": r["text"]} for r in _val_rows(d)[:3]]
    out = _run_predict(d, tmp_path, rows)
    probs = np.load(_seed_dir(d, result) / "val_preds.npy")
    np.testing.assert_allclose([o["score"] for o in out], probs[:3, 1], atol=1e-5)
    assert all("flag" not in o for o in out)


def test_exported_predict_token_crf(promoted_tok, tmp_path):
    d, _, result = promoted_tok
    assert result["status"] == "done", result
    assert not (d / "final" / "threshold.json").exists()
    rows = [{"id": r["id"], "tokens": r["tokens"]} for r in _val_rows(d)[:3]]
    out = _run_predict(d, tmp_path, rows)
    preds = json.loads((_seed_dir(d, result) / "val_preds.json").read_text())
    assert [o["tags"] for o in out] == preds[:3]


def test_wait_returns_promote_payload(promoted, runner):
    d, _, result = promoted
    r = runner.invoke(app, ["wait", result["run_id"], "--exp", str(d), "--timeout", "1"])
    assert r.exit_code == 0, r.stdout
    out = json.loads(r.stdout)
    assert out["stage"] == "promote" and Path(out["promote"]["final"]).resolve() == (d / "final").resolve()
    assert out["promote"]["test_metrics"]["n_rows"] == 12 and "throughput" in out["promote"]


def test_failure_before_test_scoring_leaves_test_unconsumed(tmp_path):
    d = _make_exp(tmp_path, objective="f1", guardrail=None)
    rid = _finished(d, "research", {"dropout": 0.2}, "keep")
    cfg = ex.load(d)[0]
    vault = vault_dir(cfg.uid) / "test.jsonl"
    vault.rename(vault.with_suffix(".bak"))
    result = promote(d, rid, seeds=1, sync=True)
    assert "error" in result and not (d / "final" / "test_metrics.json").exists()
    state = ex.load(d)[2]
    assert state.promoted is False and state.promote_pending is None
    vault.with_suffix(".bak").rename(vault)
    assert promote(d, rid, seeds=1)["status"] == "queued"  # test still unused: promote allowed again


def test_crashed_seeds_clear_pending(tmp_path):
    d = _make_exp(tmp_path, objective="f1", guardrail=None)
    rid = _finished(d, "research", {"not_a_key": 1}, "keep")  # the worker fails on an unknown key
    result = promote(d, rid, seeds=1, sync=True)
    assert result["status"] == "crash" and "error" in result
    state = ex.load(d)[2]
    assert state.promoted is False and state.promote_pending is None


def test_read_test_only_called_by_promote():
    callers = sorted(
        p.relative_to(SRC).as_posix()
        for p in SRC.rglob("*.py")
        if re.search(r"(?<!def )\bread_test\(", p.read_text())
    )
    assert callers == ["promote.py"]


def test_crash_between_test_scoring_and_state_update(tmp_path, monkeypatch):
    from finetune_kit import data
    from finetune_kit import promote as pr

    d = _make_exp(tmp_path, objective="f1", guardrail=None)
    rid = _finished(d, "research", {"dropout": 0.2}, "keep")

    def boom(exp_dir):
        raise RuntimeError("killed before state.promoted was saved")

    monkeypatch.setattr(pr, "_mark_promoted", boom)
    first = promote(d, rid, seeds=1, sync=True)
    assert "error" in first
    scored = json.loads((d / "final" / "test_metrics.json").read_text())  # written before the state
    assert scored["promote_run"] == first["run_id"] and ex.load(d)[2].promoted is False
    with pytest.raises(ValueError, match="already promoted"):  # the test counts as used
        promote(d, rid)

    monkeypatch.undo()
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(data, "read_test", lambda uid: pytest.fail("test split read twice"))
    resumed = pr.finish(d, store.read_request(d, first["run_id"]))  # what recover_stale re-runs
    assert "error" not in resumed, resumed
    assert resumed["test_metrics"] == scored and ex.load(d)[2].promoted is True
    assert (d / "final" / "predict.py").is_file()


MODELING = """\
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from .configuration_tinyremote import TinyRemoteConfig


class TinyRemoteModel(PreTrainedModel):
    config_class = TinyRemoteConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, **kw):
        return BaseModelOutput(last_hidden_state=self.embed(input_ids))
"""
CONFIGURATION = """\
from transformers import PretrainedConfig


class TinyRemoteConfig(PretrainedConfig):
    model_type = "tinyremote"

    def __init__(self, vocab_size=2000, hidden_size=8, num_hidden_layers=1, **kw):
        self.vocab_size, self.hidden_size, self.num_hidden_layers = vocab_size, hidden_size, num_hidden_layers
        super().__init__(**kw)
"""


def test_remote_code_model_saved_self_contained(tmp_path):
    """A trust_remote_code backbone exports its code files; auto_map resolves inside model/."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from finetune_kit.profiles import load_profiles
    from finetune_kit.promote import save_model
    from finetune_kit.train.config import build_config
    from finetune_kit.train.models import SeqClassifier

    repo = tmp_path / "hub" / "tinyremote"
    repo.mkdir(parents=True)
    (repo / "modeling_tinyremote.py").write_text(MODELING)
    (repo / "configuration_tinyremote.py").write_text(CONFIGURATION)
    (repo / "config.json").write_text(json.dumps({
        "model_type": "tinyremote", "vocab_size": 2000, "hidden_size": 8, "num_hidden_layers": 1,
        "auto_map": {"AutoConfig": "configuration_tinyremote.TinyRemoteConfig",
                     "AutoModel": "modeling_tinyremote.TinyRemoteModel"},
    }))
    tiny = load_profiles(None)["tiny-bert"]
    AutoTokenizer.from_pretrained(tiny.hf_id).save_pretrained(repo)
    profile = type(tiny)(**{**tiny.__dict__, "key": "tinyremote", "hf_id": str(repo), "trust_remote_code": True})
    backbone = AutoModel.from_config(AutoConfig.from_pretrained(repo, trust_remote_code=True), trust_remote_code=True)
    cfg = build_config({}, profile, "seq-cls", 0)
    model = SeqClassifier(backbone, 2, cfg)

    out = tmp_path / "model"
    save_model(model, profile, out)
    assert (out / "modeling_tinyremote.py").is_file() and (out / "configuration_tinyremote.py").is_file()
    auto_map = json.loads((out / "config.json").read_text())["auto_map"]
    assert auto_map["AutoModel"] == "modeling_tinyremote.TinyRemoteModel"
    assert auto_map["AutoConfig"] == "configuration_tinyremote.TinyRemoteConfig"
    assert not any("--" in v for v in auto_map.values())

    shutil.rmtree(repo)  # the export must not need the source repo, the hub or the module cache
    script = (
        "import sys, torch; from pathlib import Path; "
        "from finetune_kit.promote import load_model; from finetune_kit.train.config import build_config; "
        "from finetune_kit.profiles import load_profiles; "
        "p = load_profiles(None)['tiny-bert']; p.trust_remote_code = True; "
        "cfg = build_config({}, p, 'seq-cls', 0); "
        "m, tok = load_model(Path(sys.argv[1]), cfg, p, 2, 'cpu'); "
        "print(type(m.backbone).__name__)"
    )
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_MODULES_CACHE": str(tmp_path / "fresh-modules")}
    p = subprocess.run([sys.executable, "-c", script, str(out)], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip().endswith("TinyRemoteModel")
    state = __import__("torch").load(out / "classifier.pt", weights_only=True)
    assert "backbone.embed.weight" in state
