import json

import numpy as np
import pytest

from finetune_kit import commands, store
from finetune_kit import experiment as ex
from finetune_kit.cli import app
from finetune_kit.data import val_size_warning
from finetune_kit.paths import vault_dir


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("FINETUNE_KIT_TEST_PROFILES", "1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(commands, "load_tokenizer", lambda profile: _StubTok())


class _StubTok:
    def __call__(self, texts, add_special_tokens=True):
        return {"input_ids": [t.split() for t in texts]}


def _out(r):
    return json.loads(r.stdout)


def _seq_rows(n=60, classes=("0", "1")):
    return [{"id": f"s{i}", "text": f"word{i} text {i}", "label": classes[i % len(classes)]} for i in range(n)]


def _new(runner, seq_jsonl, *extra, rows=None, metric="macro_f1", name="gate"):
    seq_jsonl(rows or _seq_rows())
    return runner.invoke(app, [
        "new", name, "--task", "seq-cls", "--model", "tiny-bert", "--train", "data.jsonl",
        "--metric", metric, *extra,
    ])


def _exp_dir(tmp_path):
    return next((tmp_path / "experiments").iterdir())


def test_new_creates_experiment(runner, seq_jsonl, tmp_path):
    r = _new(runner, seq_jsonl)
    assert r.exit_code == 0, r.stdout
    assert _out(r)["experiment"] == "001-gate"
    cfg, _, _ = ex.load(_exp_dir(tmp_path))
    assert cfg.versions["model_revision"] == "main" and "finetune-kit" in cfg.versions


def test_new_rejects_bad_metric_for_task(runner, seq_jsonl):
    r = _new(runner, seq_jsonl, metric="entity_f1")
    assert r.exit_code == 2 and "not valid for seq-cls" in _out(r)["error"]


def test_new_requires_positive_class_multiclass(runner, seq_jsonl):
    rows = _seq_rows(30, ("a", "b", "c"))
    r = _new(runner, seq_jsonl, rows=rows, metric="pr_auc")
    assert r.exit_code == 2 and "--positive-class" in _out(r)["error"]
    ok = _new(runner, seq_jsonl, "--positive-class", "b", rows=rows, metric="pr_auc")
    assert ok.exit_code == 0, ok.stdout


def test_new_rejects_empty_slug_and_unsupported_model(runner, seq_jsonl):
    assert _new(runner, seq_jsonl, name="!!!").exit_code == 2
    seq_jsonl(_seq_rows())
    r = runner.invoke(app, ["new", "x", "--task", "seq-cls", "--model", "nope", "--train", "data.jsonl",
                            "--metric", "macro_f1"])
    assert r.exit_code == 2 and "unknown model" in _out(r)["error"]


def test_new_custom_metric_needs_direction(runner, seq_jsonl, tmp_path):
    (tmp_path / "m.py").write_text("def mine(y, probs):\n    return 0.5\n")
    r = _new(runner, seq_jsonl, "--custom-metric", "m.py:mine", metric="mine")
    assert r.exit_code == 2 and "--direction" in _out(r)["error"]
    ok = _new(runner, seq_jsonl, "--custom-metric", "m.py:mine", "--direction", "max", metric="mine")
    assert ok.exit_code == 0, ok.stdout


def test_models_and_metrics_listing(runner):
    r = runner.invoke(app, ["models", "--task", "token-cls"])
    assert r.exit_code == 0 and "tiny-bert" in [m["key"] for m in _out(r)["models"]]
    r = runner.invoke(app, ["metrics", "--task", "token-cls"])
    assert {m["name"] for m in _out(r)["metrics"]} == {"entity_f1", "token_accuracy"}


def _checked(runner, seq_jsonl, tmp_path, *extra, metric="macro_f1", rows=None):
    assert _new(runner, seq_jsonl, *extra, metric=metric, rows=rows).exit_code == 0
    r = runner.invoke(app, ["check-data", "001"])
    assert r.exit_code == 0, r.stdout
    return _exp_dir(tmp_path), _out(r)


def test_check_data_splits_and_vaults(runner, seq_jsonl, tmp_path):
    d, report = _checked(runner, seq_jsonl, tmp_path)
    cfg, _, state = ex.load(d)
    assert (d / "splits" / "train.jsonl").is_file() and (d / "splits" / "val.jsonl").is_file()
    assert not (d / "splits" / "test.jsonl").exists()
    assert (vault_dir(cfg.uid) / "test.jsonl").is_file()
    assert (d / "data-report.json").is_file()
    assert report["n_rows"] == {"train": 48, "val": 6, "test": 6}
    assert "splits/train.jsonl" in state.checksums


def test_check_data_refuses_resplit(runner, seq_jsonl, tmp_path):
    d, _ = _checked(runner, seq_jsonl, tmp_path)
    r = runner.invoke(app, ["check-data", "001"])
    assert r.exit_code == 2 and "--resplit" in _out(r)["error"]
    cfg, _, state = ex.load(d)
    state.runs_used = 5
    ex.save_state(d, state)
    r = runner.invoke(app, ["check-data", "001", "--resplit", "--val-frac", "0.2"])
    assert r.exit_code == 0, r.stdout
    assert _out(r)["n_rows"]["val"] == 12
    assert ex.load(d)[2].runs_used == 0


def test_check_data_reports_val_warning(runner, seq_jsonl, tmp_path):
    _, report = _checked(runner, seq_jsonl, tmp_path)
    assert any("smallest val class" in w for w in report["warnings"])
    assert val_size_warning("seq-cls", "macro_f1", {}, {"0": 3, "1": 3}, "1") in report["warnings"]


def test_check_data_recall_val_warning_uses_metric_defaults(runner, seq_jsonl, tmp_path):
    _, report = _checked(runner, seq_jsonl, tmp_path, metric="skip_at_recall")
    assert any("missed positives" in w for w in report["warnings"])


def test_catalog_marks_recommended(runner, seq_jsonl, tmp_path):
    rows = [{"id": f"s{i}", "text": f"t{i}", "label": "1" if i % 10 == 0 else "0"} for i in range(100)]
    _checked(runner, seq_jsonl, tmp_path, rows=rows)
    r = runner.invoke(app, ["catalog", "001"])
    items = {c["name"]: c for c in _out(r)["catalog"]}
    assert items["class_weight"]["recommended"] and items["lr"]["recommended"]
    assert not items["dropout"]["recommended"]


def test_space_commit_cli(runner, seq_jsonl, tmp_path):
    _new(runner, seq_jsonl)
    r = runner.invoke(app, ["space", "commit", "001", "--tunable", "lr,dropout", "--losses", "ce,focal"])
    assert r.exit_code == 0, r.stdout
    assert _out(r) == {"round": 1, "tunable": ["lr", "dropout"], "losses": ["ce", "focal"]}
    assert ex.load(_exp_dir(tmp_path))[1].losses == ["ce", "focal"]
    bad = runner.invoke(app, ["space", "commit", "001", "--tunable", "nope"])
    assert bad.exit_code == 2


def _add_result(d, run_id, mean, std, verdict, stage="research", desc="d"):
    store.append_result(d, {
        "run_id": run_id, "round": 1, "stage": stage, "objective_mean": mean, "objective_std": std,
        "guardrail_mean": None, "n_seeds": 3, "minutes": 1.5, "verdict": verdict, "description": desc,
        "config": json.dumps({"dropout": 0.2}),
    })


@pytest.fixture
def results_exp(runner, seq_jsonl, tmp_path):
    _new(runner, seq_jsonl)
    d = _exp_dir(tmp_path)
    _, _, st = ex.load(d)
    st.sigma_obj, st.best_run, st.noise_run = 0.01, "r002", "r001"
    ex.save_state(d, st)
    _add_result(d, "r001", 0.80, 0.01, "baseline", stage="noise")
    _add_result(d, "r002", 0.85, 0.01, "keep")
    _add_result(d, "r003", 0.84, 0.02, "discard")
    return d


def test_compare_stop_flag(runner, results_exp):
    out = _out(runner.invoke(app, ["compare", "001"]))
    assert out["best"] == "r002" and out["stop"] is False and out["streak"] == 0
    assert out["noise"]["sigma_obj"] == 0.01 and out["noise"]["keep_bar"] == pytest.approx(0.02 * (2 / 3) ** 0.5)
    assert out["budget_left"] == {"runs": 40, "hours": 24}
    assert [r["run_id"] for r in out["runs"]] == ["r001", "r002", "r003"]
    assert out["runs"][1]["objective_mean"] == 0.85 and out["runs"][1]["config"] == {"dropout": 0.2}
    _, _, st = ex.load(results_exp)
    st.non_keep_streak = 10
    ex.save_state(results_exp, st)
    assert _out(runner.invoke(app, ["compare", "001"]))["stop"] is True
    st.non_keep_streak, st.runs_used = 0, 40
    ex.save_state(results_exp, st)
    out = _out(runner.invoke(app, ["compare", "001"]))
    assert out["stop"] is True and out["budget_left"]["runs"] == 0


def test_report_writes_html(runner, results_exp):
    r = runner.invoke(app, ["report", "001"])
    assert r.exit_code == 0, r.stdout
    path = _out(r)["path"]
    html = open(path).read()
    assert path.endswith("report.html")
    for rid in ("r001", "r002", "r003"):
        assert rid in html
    assert "<svg" in html and "http" not in html and "keep bar" in html


# ---- misses ----


@pytest.fixture
def miss_exp(runner, seq_jsonl, tmp_path):
    _new(runner, seq_jsonl)
    d = _exp_dir(tmp_path)
    splits = d / "splits"
    splits.mkdir()
    train = [{"id": "t0", "text": "a", "label": "0"}, {"id": "t1", "text": "b", "label": "1"}]
    labels = ["1", "0", "1", "0", "1", "0", "0", "1"]
    val = [{"id": f"v{i}", "text": f"text {i} " + "x" * 400, "label": l} for i, l in enumerate(labels)]
    (splits / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    (splits / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in val))
    scores = [0.30, 0.10, 0.70, 0.98, 0.20, 0.60, 0.01, 0.90]  # P(label 1)
    run = d / "runs" / "r001"
    for seed, obj in ((0, 0.7), (1, 0.9)):
        (run / f"seed_{seed}").mkdir(parents=True)
        (run / f"seed_{seed}" / "metrics.json").write_text(json.dumps({"objective": obj}))
    probs = np.array([[1 - s, s] for s in scores])
    np.save(run / "seed_1" / "val_preds.npy", probs)
    np.save(run / "seed_0" / "val_preds.npy", np.zeros((8, 2)) + 0.5)  # worse seed, must be ignored
    (run / "request.json").write_text(json.dumps({"run_id": "r001", "stage": "research", "seeds": [0, 1]}))
    return d


def test_misses_seq_sorted(runner, miss_exp):
    out = _out(runner.invoke(app, ["misses", "001", "r001"]))
    assert out["seed"] == 1
    fns = out["false_negatives"]
    assert [i["id"] for i in fns] == ["v4", "v0"]  # label 1, p<0.5, ascending score
    assert [i["score"] for i in fns] == sorted(i["score"] for i in fns)
    assert len(fns[0]["text"]) == 300 and fns[0]["label"] == "1"
    assert [i["id"] for i in out["false_positives"]] == ["v3", "v5"]  # descending score
    k1 = _out(runner.invoke(app, ["misses", "001", "r001", "--k", "1"]))
    assert len(k1["false_negatives"]) == 1 and len(k1["false_positives"]) == 1


def test_misses_label_issue_flagged(runner, miss_exp):
    out = _out(runner.invoke(app, ["misses", "001", "r001"]))
    assert [i["id"] for i in out["possible_label_issues"]] == ["v3"]
    assert out["possible_label_issues"][0]["score"] == pytest.approx(0.98)


def test_misses_uses_constrained_threshold(runner, miss_exp):
    cfg_path = miss_exp / "experiment.yaml"
    import yaml

    raw = yaml.safe_load(cfg_path.read_text())
    raw["objective"] = {"metric": "skip_at_recall", "params": {"recall": 1.0}}
    cfg_path.write_text(yaml.safe_dump(raw))
    out = _out(runner.invoke(app, ["misses", "001", "r001"]))
    assert out["threshold"] == pytest.approx(0.2)  # lowest positive score
    assert out["false_negatives"] == []
    assert {i["id"] for i in out["false_positives"]} == {"v3", "v5"}


@pytest.fixture
def tok_exp(runner, tmp_path):
    rows = [{"id": f"t{i}", "tokens": ["a", "b", "c"], "tags": ["B-PER", "O", "O"]} for i in range(10)]
    (tmp_path / "tok.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    r = runner.invoke(app, ["new", "ner", "--task", "token-cls", "--model", "tiny-bert", "--train", "tok.jsonl",
                            "--metric", "entity_f1"])
    assert r.exit_code == 0, r.stdout
    d = _exp_dir(tmp_path)
    (d / "splits").mkdir()
    val = [
        {"id": "v0", "tokens": ["john", "went", "home"], "tags": ["B-PER", "O", "O"]},
        {"id": "v1", "tokens": ["mary", "smith", "ran"], "tags": ["B-PER", "I-PER", "O"]},
        {"id": "v2", "tokens": ["bob", "ran", "far"], "tags": ["B-PER", "O", "B-ORG"]},
    ]
    (d / "splits" / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in val))
    preds = [["B-PER", "O", "O"], ["B-PER", "O", "O"], ["O", "O", "B-LOC"]]
    run = d / "runs" / "r001" / "seed_0"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"objective": 0.5}))
    (run / "val_preds.json").write_text(json.dumps(preds))
    (d / "runs" / "r001" / "request.json").write_text(json.dumps({"run_id": "r001", "seeds": [0]}))
    return d


def test_misses_token_spans(runner, tok_exp):
    out = _out(runner.invoke(app, ["misses", "001", "r001"]))
    rows = out["rows"]
    assert [r["id"] for r in rows] == ["v2", "v1"]  # most errors first; v0 is exact
    assert rows[0]["gold"] == [["PER", 0, 1, "bob"], ["ORG", 2, 3, "far"]] and rows[0]["pred"] == [["LOC", 2, 3, "far"]]
    assert rows[1]["gold"] == [["PER", 0, 2, "mary smith"]] and rows[1]["pred"] == [["PER", 0, 1, "mary"]]
    assert len(_out(runner.invoke(app, ["misses", "001", "r001", "--k", "1"]))["rows"]) == 1


def test_check_data_tokenizer_failure_leaves_nothing_frozen(runner, seq_jsonl, tmp_path, monkeypatch):
    _new(runner, seq_jsonl)
    d = _exp_dir(tmp_path)
    cfg, _, _ = ex.load(d)

    def boom(profile):
        raise ValueError("offline")

    monkeypatch.setattr(commands, "load_tokenizer", boom)
    r = runner.invoke(app, ["check-data", "001"])
    assert r.exit_code == 2 and "offline" in _out(r)["error"]
    assert not (d / "splits").exists() and not (vault_dir(cfg.uid) / "test.jsonl").exists()
    monkeypatch.setattr(commands, "load_tokenizer", lambda profile: _StubTok())
    assert runner.invoke(app, ["check-data", "001"]).exit_code == 0
