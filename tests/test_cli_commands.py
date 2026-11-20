import json

import numpy as np
import pytest

from finetune_kit import commands, sources, store
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
    # allowing a loss besides ce makes `loss` itself tunable
    assert _out(r) == {"round": 1, "tunable": ["lr", "dropout", "loss"], "losses": ["ce", "focal"]}
    assert ex.load(_exp_dir(tmp_path))[1].losses == ["ce", "focal"]
    bad = runner.invoke(app, ["space", "commit", "001", "--tunable", "nope", "--losses", "ce"])
    assert bad.exit_code == 2


def test_space_commit_requires_losses(runner, seq_jsonl, tmp_path):
    _new(runner, seq_jsonl)
    r = runner.invoke(app, ["space", "commit", "001", "--tunable", "lr"])
    assert r.exit_code == 2
    assert "--losses is required" in _out(r)["error"]
    assert not (_exp_dir(tmp_path) / "search-space.yaml").exists()


def test_space_commit_always_allows_ce(runner, seq_jsonl, tmp_path):
    _new(runner, seq_jsonl)
    r = runner.invoke(app, ["space", "commit", "001", "--tunable", "lr", "--losses", "focal"])
    assert r.exit_code == 0, r.stdout
    assert _out(r) == {"round": 1, "tunable": ["lr", "loss"], "losses": ["ce", "focal"]}


def test_space_commit_ce_only_keeps_loss_fixed(runner, seq_jsonl, tmp_path):
    _new(runner, seq_jsonl)
    r = runner.invoke(app, ["space", "commit", "001", "--tunable", "lr", "--losses", "ce"])
    assert _out(r) == {"round": 1, "tunable": ["lr"], "losses": ["ce"]}


def test_catalog_lists_losses_with_recommendations(runner, seq_jsonl, tmp_path):
    rows = [{"id": f"s{i}", "text": f"t{i}", "label": "1" if i % 10 == 0 else "0"} for i in range(100)]
    _checked(runner, seq_jsonl, tmp_path, rows=rows)
    out = _out(runner.invoke(app, ["catalog", "001"]))
    losses = {l["name"]: l for l in out["losses"]}
    assert list(losses) == ["ce", "focal", "asl", "rank", "ce+rank"]
    assert all(l["help"] for l in losses.values())
    assert losses["focal"]["recommended"] and "minority class" in losses["focal"]["reason"]
    assert not losses["ce"]["recommended"] and losses["ce"]["always"] is True
    assert losses["rank"]["always"] is False


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


@pytest.mark.parametrize("exc", [OSError("offline: model not cached"), OverflowError("too big")])
def test_check_data_os_and_overflow_errors_are_json(runner, seq_jsonl, tmp_path, monkeypatch, exc):
    _new(runner, seq_jsonl)

    def boom(profile):
        raise exc

    monkeypatch.setattr(commands, "load_tokenizer", boom)
    r = runner.invoke(app, ["check-data", "001"])
    assert r.exit_code == 2 and str(exc) in _out(r)["error"]


def test_check_data_refuses_empty_val_and_freezes_nothing(runner, seq_jsonl, tmp_path):
    rows = _seq_rows()
    seq_jsonl(rows[:40], "tr.jsonl")
    seq_jsonl([], "va.jsonl")
    seq_jsonl(rows[40:], "te.jsonl")
    r = runner.invoke(app, ["new", "g", "--task", "seq-cls", "--model", "tiny-bert", "--train", "tr.jsonl",
                            "--val", "va.jsonl", "--test", "te.jsonl", "--metric", "macro_f1"])
    assert r.exit_code == 0, r.stdout
    r = runner.invoke(app, ["check-data", "001"])
    assert r.exit_code == 2 and "val split is empty" in _out(r)["error"]
    assert not (_exp_dir(tmp_path) / "splits").exists()


@pytest.mark.parametrize("split", ["val", "test"])
def test_check_data_refuses_unseen_seq_labels(runner, seq_jsonl, tmp_path, split):
    rows = _seq_rows()
    other = [{"id": f"x{i}", "text": f"other {i}", "label": "0"} for i in range(10)]
    unseen = [{"id": "u0", "text": "new class", "label": "9"}]
    seq_jsonl(rows, "tr.jsonl")
    seq_jsonl(other + (unseen if split == "val" else []), "va.jsonl")
    seq_jsonl(other[:5] + (unseen if split == "test" else []), "te.jsonl")
    r = runner.invoke(app, ["new", "g", "--task", "seq-cls", "--model", "tiny-bert", "--train", "tr.jsonl",
                            "--val", "va.jsonl", "--test", "te.jsonl", "--metric", "macro_f1"])
    assert r.exit_code == 0, r.stdout
    r = runner.invoke(app, ["check-data", "001"])
    assert r.exit_code == 2 and "'9'" in _out(r)["error"]
    assert not (_exp_dir(tmp_path) / "splits").exists()


def test_check_data_refuses_unseen_token_tags(runner, tok_jsonl, tmp_path):
    train = [{"id": f"t{i}", "tokens": ["a", "Ann"], "tags": ["O", "B-PER"]} for i in range(20)]
    val = [{"id": "v0", "tokens": ["a", "Rome"], "tags": ["O", "B-LOC"]}]
    test = [{"id": "e0", "tokens": ["a", "Ann"], "tags": ["O", "B-PER"]}]
    tok_jsonl(train, "tr.jsonl")
    tok_jsonl(val, "va.jsonl")
    tok_jsonl(test, "te.jsonl")
    r = runner.invoke(app, ["new", "n", "--task", "token-cls", "--model", "tiny-bert", "--train", "tr.jsonl",
                            "--val", "va.jsonl", "--test", "te.jsonl", "--metric", "entity_f1"])
    assert r.exit_code == 0, r.stdout
    r = runner.invoke(app, ["check-data", "001"])
    assert r.exit_code == 2 and "LOC" in _out(r)["error"]
    assert not (_exp_dir(tmp_path) / "splits").exists()


def test_check_data_resplit_refused_after_noise_run(runner, seq_jsonl, tmp_path):
    d, _ = _checked(runner, seq_jsonl, tmp_path)
    store.new_run(d, "noise", {}, [0, 1, 2], "noise floor")
    before = ex.load(d)[2]
    r = runner.invoke(app, ["check-data", "001", "--resplit"])
    assert r.exit_code == 2 and "splits are baselined; start a new experiment" in _out(r)["error"]
    assert ex.load(d)[2] == before and (d / "splits" / "train.jsonl").is_file()


def test_new_requires_two_seeds(runner, seq_jsonl):
    r = _new(runner, seq_jsonl, "--seeds", "1")
    assert r.exit_code == 2 and "--seeds must be at least 2" in _out(r)["error"]


def test_new_with_csv_and_columns(runner, tmp_path):
    (tmp_path / "train.csv").write_text(
        "prompt,is_injection\n" + "".join(f"msg {i} here,{i % 2}\n" for i in range(60)), encoding="utf-8"
    )
    r = runner.invoke(app, [
        "new", "csvexp", "--task", "seq-cls", "--model", "tiny-bert", "--train", "train.csv",
        "--metric", "macro_f1", "--columns", '{"text":"prompt","label":"is_injection"}',
    ])
    assert r.exit_code == 0, r.stdout
    exp = tmp_path / "experiments" / _out(r)["experiment"]
    assert ex.load(exp)[0].data.columns == {"text": "prompt", "label": "is_injection"}
    r = runner.invoke(app, ["check-data", exp.name[:3]])
    assert r.exit_code == 0, r.stdout
    frozen = [json.loads(l) for l in (exp / "splits" / "train.jsonl").read_text().splitlines()]
    assert frozen and all(set(x) == {"id", "text", "label"} for x in frozen)
    assert frozen[0]["id"].startswith("train-")


def test_new_bad_column_writes_nothing(runner, tmp_path):
    (tmp_path / "train.csv").write_text("prompt,label\nhi,0\nyo,1\n", encoding="utf-8")
    r = runner.invoke(app, [
        "new", "csvexp", "--task", "seq-cls", "--model", "tiny-bert", "--train", "train.csv",
        "--metric", "macro_f1", "--columns", '{"text":"promt"}',
    ])
    assert r.exit_code == 2
    assert "no column 'promt'" in _out(r)["error"]
    assert not list((tmp_path / "experiments").glob("001-*"))


# ---- Hugging Face sources ----

def _hf_rows(n, tag):
    return [{"text": f"{tag} word{i} text", "label": i % 2} for i in range(n)]


def _hf_new(runner, *extra, name="hf"):
    return runner.invoke(app, [
        "new", name, "--task", "seq-cls", "--model", "tiny-bert", "--source", "hf:x/y",
        "--metric", "macro_f1", *extra,
    ])


def _hf_published(hf_fake):
    hf_fake({"train": _hf_rows(40, "tr"), "validation": _hf_rows(8, "va"), "test": _hf_rows(6, "te")})


def test_new_from_hf_pins_revision(runner, tmp_path, hf_fake):
    _hf_published(hf_fake)
    r = _hf_new(runner, "--revision", "main")
    assert r.exit_code == 0, r.stdout
    cfg, _, _ = ex.load(_exp_dir(tmp_path))
    assert cfg.data.source == "hf:x/y" and cfg.data.revision == "abc123" and cfg.data.train is None


def test_check_data_from_hf_freezes_published_splits(runner, tmp_path, hf_fake):
    _hf_published(hf_fake)
    assert _hf_new(runner).exit_code == 0
    r = runner.invoke(app, ["check-data", "001"])
    assert r.exit_code == 0, r.stdout
    report, d = _out(r), _exp_dir(tmp_path)
    cfg, _, _ = ex.load(d)
    assert len((d / "splits" / "train.jsonl").read_text().splitlines()) == 40
    assert len((vault_dir(cfg.uid) / "test.jsonl").read_text().splitlines()) == 6
    assert report["split_origin"] == {"train": "published", "val": "published", "test": "published"}
    assert report["ignored_splits"] == []
    assert not any("carved" in w for w in report["warnings"])


def test_check_data_from_hf_carved_warnings_and_ignored(runner, tmp_path, hf_fake):
    hf_fake({"train": _hf_rows(100, "tr"), "unsupervised": _hf_rows(3, "un")})
    assert _hf_new(runner).exit_code == 0
    report = _out(runner.invoke(app, ["check-data", "001"]))
    assert report["ignored_splits"] == ["unsupervised"]
    assert "val carved from train: no published split" in report["warnings"]
    assert "test carved from train: no published test split" in report["warnings"]


def test_check_data_local_single_file_warning(runner, seq_jsonl, tmp_path):
    _, report = _checked(runner, seq_jsonl, tmp_path)
    assert report["split_origin"] == {"train": "provided", "val": "carved from train", "test": "carved from train"}
    assert "val carved from train: single file given" in report["warnings"]


def test_hf_not_found_is_json_error(runner, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise ValueError("could not load hf:x/y: not found")

    monkeypatch.setattr(sources, "_resolve_revision", boom)
    r = _hf_new(runner)
    assert r.exit_code == 2
    assert "could not load" in _out(r)["error"]
    assert not (tmp_path / "experiments").exists() or not list((tmp_path / "experiments").iterdir())


def test_hf_load_failure_leaves_no_experiment(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "_resolve_revision", lambda repo, rev, cfg=None: "abc123")

    def boom(*a, **k):
        raise ValueError("could not load hf:x/y: gone")

    monkeypatch.setattr(sources, "_load_dataset", boom)
    r = _hf_new(runner)
    assert r.exit_code == 2 and "could not load" in _out(r)["error"]
    assert not (tmp_path / "experiments").exists() or not list((tmp_path / "experiments").iterdir())


def test_train_and_source_both_refused(runner, seq_jsonl, tmp_path, hf_fake):
    _hf_published(hf_fake)
    r = _new(runner, seq_jsonl, "--source", "hf:x/y")
    assert r.exit_code == 2 and _out(r) == {"error": "give --train or --source, not both"}


def test_neither_train_nor_source_refused(runner):
    r = runner.invoke(app, ["new", "n", "--task", "seq-cls", "--model", "tiny-bert", "--metric", "macro_f1"])
    assert r.exit_code == 2 and _out(r) == {"error": "give --train or --source"}


def test_source_with_val_or_revision_without_source_refused(runner, seq_jsonl, hf_fake):
    _hf_published(hf_fake)
    r = _hf_new(runner, "--val", "v.jsonl", "--test", "t.jsonl")
    assert r.exit_code == 2 and _out(r) == {"error": "--val and --test only apply with --train"}
    r = _new(runner, seq_jsonl, "--revision", "main")
    assert r.exit_code == 2 and _out(r) == {"error": "--revision only applies with --source"}


def test_bad_hf_spec_is_one_line_error(runner):
    r = runner.invoke(app, ["new", "n", "--task", "seq-cls", "--model", "tiny-bert", "--source", "glue",
                            "--metric", "macro_f1"])
    assert r.exit_code == 2 and "Hugging Face source must be" in _out(r)["error"]


def test_hf_positive_class_checked_against_named_labels(runner, hf_fake):
    _hf_published(hf_fake)
    r = _hf_new(runner, "--positive-class", "injection")
    assert r.exit_code == 0, r.stdout
    r = _hf_new(runner, "--positive-class", "1", name="other")
    assert r.exit_code == 2 and "not a train label" in _out(r)["error"]


def test_existing_jsonl_experiment_yaml_still_loads(runner, seq_jsonl, tmp_path):
    assert _new(runner, seq_jsonl).exit_code == 0
    d = _exp_dir(tmp_path)
    import yaml

    raw = yaml.safe_load((d / "experiment.yaml").read_text())
    for k in ("source", "revision"):
        raw["data"].pop(k, None)
    assert set(raw["data"]) <= {"train", "val", "test", "columns"}
    cfg = ex.ExperimentConfig(**raw)
    assert cfg.data.train == "data.jsonl" and cfg.data.source is None


def test_cli_inspect_data_json(runner, tmp_path):
    (tmp_path / "t.csv").write_text("prompt,is_injection\n" + "hello,0\nbye,1\n" * 3, encoding="utf-8")
    r = runner.invoke(app, ["inspect-data", "--train", "t.csv"])
    assert r.exit_code == 0, r.stdout
    out = _out(r)
    assert out["suggested_columns"] == {"text": "prompt", "label": "is_injection"}
    assert "hello" not in r.stdout


def test_cli_inspect_data_hf_and_error(runner, hf_fake):
    _hf_published(hf_fake)
    out = _out(runner.invoke(app, ["inspect-data", "--source", "hf:x/y"]))
    assert out["revision"] == "abc123" and out["splits"]["validation"]["role"] == "val"
    r = runner.invoke(app, ["inspect-data"])
    assert r.exit_code == 2 and _out(r) == {"error": "give --train or --source"}


# ---- final-review fixes ----

def test_new_token_cls_bad_mapping_writes_nothing(runner, tok_jsonl, tmp_path):
    rows = [{"id": f"t{i}", "words": ["a", "Ann"], "tags": ["O", "B-PER"]} for i in range(20)]
    tok_jsonl(rows, "tr.jsonl")
    r = runner.invoke(app, ["new", "n", "--task", "token-cls", "--model", "tiny-bert", "--train", "tr.jsonl",
                            "--metric", "entity_f1", "--columns", '{"tokens":"wrds"}'])
    assert r.exit_code == 2 and "no column 'wrds'" in _out(r)["error"]
    assert not (tmp_path / "experiments").exists() or not list((tmp_path / "experiments").iterdir())


def test_new_bad_mapping_in_local_test_file_writes_nothing(runner, tmp_path):
    (tmp_path / "tr.csv").write_text("prompt,y\n" + "".join(f"m {i},{i % 2}\n" for i in range(20)))
    (tmp_path / "va.csv").write_text("prompt,y\nm,0\nn,1\n")
    (tmp_path / "te.csv").write_text("text,y\nm,0\nn,1\n")
    r = runner.invoke(app, ["new", "n", "--task", "seq-cls", "--model", "tiny-bert", "--train", "tr.csv",
                            "--val", "va.csv", "--test", "te.csv", "--metric", "macro_f1",
                            "--columns", '{"text":"prompt","label":"y"}'])
    assert r.exit_code == 2 and "te.csv: no column 'prompt'" in _out(r)["error"]
    assert not (tmp_path / "experiments").exists() or not list((tmp_path / "experiments").iterdir())


def _text_as_label_csv(path, secret):
    path.write_text("prompt,y\n" + "".join(f"{secret} {i},{i % 2}\n" for i in range(60)))


def test_new_refuses_text_column_as_label(runner, tmp_path):
    secret = "SENTINEL-row-text-zq81 a sentence well over forty characters"
    _text_as_label_csv(tmp_path / "tr.csv", secret)
    r = runner.invoke(app, ["new", "n", "--task", "seq-cls", "--model", "tiny-bert", "--train", "tr.csv",
                            "--metric", "macro_f1", "--positive-class", "x",
                            "--columns", '{"text":"y","label":"prompt"}'])
    assert r.exit_code == 2
    assert secret not in r.stdout
    assert "is --columns mapping the text column as the label?" in _out(r)["error"]
    assert not (tmp_path / "experiments").exists() or not list((tmp_path / "experiments").iterdir())


def test_check_data_refuses_text_column_as_label(runner, tmp_path):
    secret = "SENTINEL-row-text-zq81-but-longer-than-forty-characters"
    (tmp_path / "tr.csv").write_text("prompt,y\n" + "".join(f"msg {i},{i % 2}\n" for i in range(60)))
    r = runner.invoke(app, ["new", "n", "--task", "seq-cls", "--model", "tiny-bert", "--train", "tr.csv",
                            "--metric", "macro_f1", "--columns", '{"text":"prompt","label":"y"}'])
    assert r.exit_code == 0, r.stdout
    rows = [f"msg {i},{i % 2}" for i in range(60)]
    rows[5] = f"msg 5,{secret}"
    (tmp_path / "tr.csv").write_text("prompt,y\n" + "\n".join(rows) + "\n")
    r = runner.invoke(app, ["check-data", "001"])
    assert r.exit_code == 2 and secret not in r.stdout
    assert "label column has" in _out(r)["error"]
    assert not (_exp_dir(tmp_path) / "splits").exists()


def test_new_id_null_mapping_saved(runner, tmp_path):
    (tmp_path / "tr.csv").write_text("id,text,label\n" + "".join(f"1,m {i},{i % 2}\n" for i in range(60)))
    r = runner.invoke(app, ["new", "n", "--task", "seq-cls", "--model", "tiny-bert", "--train", "tr.csv",
                            "--metric", "macro_f1", "--columns", '{"id": null}'])
    assert r.exit_code == 0, r.stdout
    assert ex.load(_exp_dir(tmp_path))[0].data.columns == {"id": None}
    assert runner.invoke(app, ["check-data", "001"]).exit_code == 0


# ---- next ----
def _next(runner, *args):
    r = runner.invoke(app, ["next", *args])
    assert r.exit_code == 0, r.stdout
    return _out(r)


def _set_state(d, **kw):
    cfg, space, state = ex.load(d)
    ex.save_state(d, state.model_copy(update=kw))


def _splits(d):
    (d / "splits").mkdir(exist_ok=True)
    (d / "splits" / "train.jsonl").write_text("{}\n")


def _space(d):
    (d / "search-space.yaml").write_text("round: 1\ntunable: [lr]\nlosses: [ce]\n")


def test_next_no_experiment_is_setup(runner, tmp_path):
    out = _next(runner)
    assert out == {"experiment": None, "stage": "choose-task", "reason": "no experiment yet", "alternatives": []}
    (tmp_path / "experiments").mkdir()
    assert _next(runner)["stage"] == "choose-task"


def test_next_order_check_then_hparams(runner, seq_jsonl, tmp_path):
    assert _new(runner, seq_jsonl).exit_code == 0
    d = _exp_dir(tmp_path)
    out = _next(runner)
    assert out["experiment"] == "001-gate" and out["stage"] == "check-data"
    assert out["reason"] == "data not checked and split yet"
    _splits(d)
    out = _next(runner, "001")
    assert out["stage"] == "choose-loss"


def test_next_bakeoff_before_hparams_until_locked(runner, seq_jsonl, tmp_path):
    assert _new(runner, seq_jsonl, "--bakeoff", "tiny-bert,tiny-roberta").exit_code == 0
    d = _exp_dir(tmp_path)
    _splits(d)
    out = _next(runner)
    assert out["stage"] == "compare-models"
    assert out["reason"] == "a comparison was requested and no model is chosen yet"
    _set_state(d, model_locked=True)
    assert _next(runner)["stage"] == "choose-loss"


def test_next_baseline_then_research(runner, seq_jsonl, tmp_path):
    assert _new(runner, seq_jsonl).exit_code == 0
    d = _exp_dir(tmp_path)
    _splits(d)
    _space(d)
    out = _next(runner)
    assert out["stage"] == "choose-learning-rate"
    _set_state(d, sigma_obj=0.01, round=1)
    out = _next(runner)
    assert out["stage"] == "start-experiment" and out["reason"] == "research round 1 is still running"
    (d / "log.md").write_text("# log\n## Summary (round 1)\nok\n")
    assert _next(runner)["stage"] == "report"


def test_next_research_done_then_analyze_with_alternative(runner, seq_jsonl, tmp_path):
    assert _new(runner, seq_jsonl).exit_code == 0
    d = _exp_dir(tmp_path)
    _splits(d)
    _space(d)
    _set_state(d, sigma_obj=0.01, round=1)
    (d / "log.md").write_text("## Summary\n")
    out = _next(runner)
    assert out["stage"] == "report" and out["reason"] == "results not reported yet"
    assert out["alternatives"] == [{"stage": "choose-loss", "reason": "try other losses or settings in another round"}]
    import os

    (d / "report.md").write_text("r")
    t = (d / "log.md").stat().st_mtime
    os.utime(d / "report.md", (t - 10, t - 10))
    assert _next(runner)["stage"] == "report"
    os.utime(d / "report.md", (t + 10, t + 10))
    out = _next(runner)
    assert out["stage"] == "finish" and out["reason"] == "final model not trained and scored yet"
    assert out["alternatives"] == [{"stage": "choose-loss", "reason": "try other losses or settings in another round"}]


def test_next_new_round_returns_research(runner, seq_jsonl, tmp_path):
    assert _new(runner, seq_jsonl).exit_code == 0
    d = _exp_dir(tmp_path)
    _splits(d)
    _space(d)
    _set_state(d, sigma_obj=0.01, round=2)
    (d / "log.md").write_text("## Summary\n")
    (d / "report.md").write_text("r")
    out = _next(runner)
    assert out["stage"] == "start-experiment" and out["reason"] == "research round 2 is still running"
    (d / "log.md").write_text("## Summary (round 2)\n")
    import os

    os.utime(d / "report.md", (1, 1))
    assert _next(runner)["stage"] == "report"


def test_next_promote_pending_then_done(runner, seq_jsonl, tmp_path):
    assert _new(runner, seq_jsonl).exit_code == 0
    d = _exp_dir(tmp_path)
    _splits(d)
    _space(d)
    _set_state(d, sigma_obj=0.01, round=1, promote_pending="r009")
    (d / "log.md").write_text("## Summary\n")
    (d / "report.md").write_text("r")
    out = _next(runner)
    assert out["stage"] == "finish" and out["reason"] == "the final model is being trained and scored"
    _set_state(d, promote_pending=None, promoted=True)
    out = _next(runner)
    assert out["stage"] == "choose-task" and out["reason"] == "this experiment is complete; start a new one"


def test_new_bakeoff_validated_and_saved(runner, seq_jsonl, tmp_path):
    for bad in ("tiny-bert", "tiny-roberta,tiny-bert", "tiny-bert,nope", "tiny-bert,tiny-roberta,tiny-bert,tiny-roberta"):
        r = _new(runner, seq_jsonl, "--bakeoff", bad)
        assert r.exit_code == 2, bad
        assert _out(r)["error"] == "--bakeoff needs 2-3 models for seq-cls, the first being --model"
    assert not (tmp_path / "experiments").exists()
    assert _new(runner, seq_jsonl, "--bakeoff", "tiny-bert,tiny-roberta").exit_code == 0
    assert ex.load(_exp_dir(tmp_path))[0].bakeoff == ["tiny-bert", "tiny-roberta"]


def test_old_yaml_without_bakeoff_loads(runner, seq_jsonl, tmp_path):
    assert _new(runner, seq_jsonl).exit_code == 0
    d = _exp_dir(tmp_path)
    import yaml

    raw = yaml.safe_load((d / "experiment.yaml").read_text())
    raw.pop("bakeoff", None)
    (d / "experiment.yaml").write_text(yaml.safe_dump(raw))
    assert ex.load(d)[0].bakeoff is None


def test_next_picks_highest_numeric_prefix_and_ignores_other_folders(runner, seq_jsonl, tmp_path):
    assert _new(runner, seq_jsonl).exit_code == 0
    d = _exp_dir(tmp_path)
    root = tmp_path / "experiments"
    (root / "archive").mkdir()
    (root / "zzz-notes").mkdir()
    import shutil

    shutil.copytree(d, root / "999-x")
    shutil.copytree(d, root / "1000-y")
    _splits(root / "1000-y")
    out = _next(runner)
    assert out["experiment"] == "1000-y" and out["stage"] == "choose-loss"


# ---- step by step: new --task, set data / model / goal ----


def _draft(runner, name="gate"):
    return runner.invoke(app, ["new", name, "--task", "seq-cls"])


def _cfg_of(tmp_path):
    return ex.load(_exp_dir(tmp_path))[0]


def test_new_task_only_writes_a_draft(runner, tmp_path):
    r = _draft(runner)
    assert r.exit_code == 0, r.stdout
    assert _out(r)["experiment"] == "001-gate"
    cfg = _cfg_of(tmp_path)
    assert cfg.model is None and cfg.data is None and cfg.objective is None


def test_set_data_model_goal_happy_path(runner, seq_jsonl, tmp_path):
    _draft(runner)
    seq_jsonl(_seq_rows())
    r = runner.invoke(app, ["set", "data", "001", "--train", "data.jsonl"])
    assert r.exit_code == 0, r.stdout
    assert _out(r)["data"]["train"] == "data.jsonl"
    r = runner.invoke(app, ["set", "model", "001", "tiny-bert"])
    assert r.exit_code == 0, r.stdout
    assert _out(r)["model"] == "tiny-bert"
    r = runner.invoke(app, ["set", "goal", "001", "--metric", "macro_f1", "--guardrail", "accuracy"])
    assert r.exit_code == 0, r.stdout
    assert _out(r)["objective"]["metric"] == "macro_f1"
    cfg = _cfg_of(tmp_path)
    assert cfg.data.train == "data.jsonl" and cfg.model == "tiny-bert"
    assert cfg.objective.metric == "macro_f1" and cfg.guardrail.metric == "accuracy"
    assert cfg.versions["model_revision"] == "main" and ex.missing_setup(cfg) is None


def test_set_model_with_compare_stores_bakeoff(runner, tmp_path):
    _draft(runner)
    r = runner.invoke(app, ["set", "model", "001", "tiny-bert", "--compare", "tiny-bert,tiny-roberta"])
    assert r.exit_code == 0, r.stdout
    assert _cfg_of(tmp_path).bakeoff == ["tiny-bert", "tiny-roberta"]


def test_set_validation_errors_write_nothing(runner, seq_jsonl, tmp_path):
    _draft(runner)
    before = (_exp_dir(tmp_path) / "experiment.yaml").read_text()
    r = runner.invoke(app, ["set", "data", "001", "--train", "missing.jsonl"])
    assert r.exit_code == 2 and "not found" in _out(r)["error"]
    r = runner.invoke(app, ["set", "data", "001"])
    assert r.exit_code == 2 and "--train or --source" in _out(r)["error"]
    r = runner.invoke(app, ["set", "model", "001", "nope"])
    assert r.exit_code == 2 and "unknown model" in _out(r)["error"]
    r = runner.invoke(app, ["set", "goal", "001", "--metric", "macro_f1"])
    assert r.exit_code == 2 and _out(r)["error"] == "choose the data first (step: choose-data)"
    seq_jsonl(_seq_rows(30, ("a", "b", "c")))
    assert runner.invoke(app, ["set", "data", "001", "--train", "data.jsonl"]).exit_code == 0
    mid = (_exp_dir(tmp_path) / "experiment.yaml").read_text()
    r = runner.invoke(app, ["set", "goal", "001", "--metric", "entity_f1"])
    assert r.exit_code == 2 and "not valid for seq-cls" in _out(r)["error"]
    r = runner.invoke(app, ["set", "goal", "001", "--metric", "pr_auc"])
    assert r.exit_code == 2 and "--positive-class" in _out(r)["error"]
    r = runner.invoke(app, ["set", "goal", "001", "--metric", "pr_auc", "--positive-class", "zzz"])
    assert r.exit_code == 2 and "not a train label" in _out(r)["error"]
    assert (_exp_dir(tmp_path) / "experiment.yaml").read_text() == mid != before
    assert _cfg_of(tmp_path).objective is None and _cfg_of(tmp_path).model is None


def test_set_refused_after_check(runner, seq_jsonl, tmp_path):
    _checked(runner, seq_jsonl, tmp_path)
    for args, what in (
        (["data", "001", "--train", "data.jsonl"], "data"),
        (["model", "001", "tiny-bert"], "model"),
        (["goal", "001", "--metric", "macro_f1"], "goal"),
    ):
        r = runner.invoke(app, ["set", *args])
        assert r.exit_code == 2, r.stdout
        assert _out(r)["error"] == f"{what} is checked and locked; start a new experiment to change it"


def test_check_data_on_draft_refused_and_writes_nothing(runner, tmp_path):
    _draft(runner)
    d = _exp_dir(tmp_path)
    r = runner.invoke(app, ["check-data", "001"])
    assert r.exit_code == 2
    assert _out(r)["error"] == "choose the data first (step: choose-data)"
    assert not (d / "splits").exists() and not (d / "data-report.json").exists()


@pytest.mark.parametrize("cmdline", [
    ["catalog", "001"], ["lr-scan", "001"], ["run", "001"], ["compare", "001"], ["report", "001"],
    ["misses", "001", "r001"], ["promote", "001", "r001"], ["bakeoff", "001", "--models", "a,b"],
    ["space", "commit", "001", "--tunable", "lr", "--losses", "ce"],
])
def test_commands_needing_setup_refuse_a_draft(runner, cmdline):
    _draft(runner)
    r = runner.invoke(app, cmdline)
    assert r.exit_code == 2, r.stdout
    assert _out(r)["error"].endswith("first (step: choose-data)")


def test_next_does_not_crash_on_draft(runner):
    _draft(runner)
    r = runner.invoke(app, ["next", "001"])
    assert r.exit_code == 0, r.stdout


def test_one_shot_new_unchanged(runner, seq_jsonl, tmp_path):
    r = _new(runner, seq_jsonl)
    assert r.exit_code == 0, r.stdout
    cfg = _cfg_of(tmp_path)
    assert ex.missing_setup(cfg) is None and cfg.objective.metric == "macro_f1"
    # task alone with partial one-shot options is not a silent draft
    r = runner.invoke(app, ["new", "x", "--task", "seq-cls", "--model", "tiny-bert"])
    assert r.exit_code == 2


# ---- choices that can change per round: set losses / learning-rate / experiments ----


def _state_of(tmp_path):
    return ex.load(_exp_dir(tmp_path))[2]


def test_set_losses_stores_in_state_not_yaml(runner, seq_jsonl, tmp_path):
    d, _ = _checked(runner, seq_jsonl, tmp_path)
    before = (d / "experiment.yaml").read_text()
    r = runner.invoke(app, ["set", "losses", "001", "focal"])
    assert r.exit_code == 0, r.stdout
    assert _out(r) == {"losses": ["ce", "focal"]}
    assert _state_of(tmp_path).chosen_losses == ["ce", "focal"]
    assert (d / "experiment.yaml").read_text() == before


def test_set_losses_errors(runner, seq_jsonl, tmp_path):
    _draft(runner)
    r = runner.invoke(app, ["set", "losses", "001", "ce"])
    assert r.exit_code == 2 and "choose the data first" in _out(r)["error"]
    _new(runner, seq_jsonl, name="full")
    r = runner.invoke(app, ["set", "losses", "002", "ce"])
    assert r.exit_code == 2 and _out(r)["error"] == "check the data first (step: check-data)"
    runner.invoke(app, ["check-data", "002"])
    r = runner.invoke(app, ["set", "losses", "002", "nonsense"])
    assert r.exit_code == 2 and "not allowed" in _out(r)["error"]
    assert ex.load(tmp_path / "experiments" / "002-full")[2].chosen_losses is None


def test_set_losses_refused_while_research_pending(runner, seq_jsonl, tmp_path):
    d, _ = _checked(runner, seq_jsonl, tmp_path)
    store.new_run(d, "research", {}, [0, 1, 2], "try")
    r = runner.invoke(app, ["set", "losses", "001", "focal"])
    assert r.exit_code == 2 and "research" in _out(r)["error"]
    assert _state_of(tmp_path).chosen_losses is None


def test_space_commit_uses_chosen_losses(runner, seq_jsonl, tmp_path):
    _checked(runner, seq_jsonl, tmp_path)
    runner.invoke(app, ["set", "losses", "001", "focal"])
    r = runner.invoke(app, ["space", "commit", "001", "--tunable", "lr"])
    assert r.exit_code == 0, r.stdout
    assert _out(r) == {"round": 1, "tunable": ["lr", "loss"], "losses": ["ce", "focal"]}


def test_space_commit_explicit_losses_win(runner, seq_jsonl, tmp_path):
    _checked(runner, seq_jsonl, tmp_path)
    runner.invoke(app, ["set", "losses", "001", "focal"])
    r = runner.invoke(app, ["space", "commit", "001", "--tunable", "lr", "--losses", "ce"])
    assert _out(r) == {"round": 1, "tunable": ["lr"], "losses": ["ce"]}


def test_round_two_losses_leave_checksums_passing(runner, seq_jsonl, tmp_path):
    d, _ = _checked(runner, seq_jsonl, tmp_path)
    root = store.project_root(d)
    assert runner.invoke(app, ["set", "losses", "001", "ce"]).exit_code == 0
    assert runner.invoke(app, ["space", "commit", "001", "--tunable", "lr"]).exit_code == 0
    store.new_run(d, "noise", {}, [0, 1, 2], "noise floor")
    ex.record_baseline_checksums(d, store.profile_for(ex.load(d)[0], {}, root))
    with ex.state_lock(d):
        st = ex.load(d)[2]
        st.sigma_obj = 0.01
        ex.save_state(d, st)
    r = runner.invoke(app, ["set", "losses", "001", "ce,focal"])
    assert r.exit_code == 0, r.stdout
    assert _state_of(tmp_path).round_open is True
    r = runner.invoke(app, ["space", "commit", "001", "--tunable", "lr,dropout"])
    assert r.exit_code == 0, r.stdout
    assert _out(r)["round"] == 2 and _out(r)["losses"] == ["ce", "focal"]
    assert _state_of(tmp_path).round_open is False
    ex.verify_checksums(d, root)


def test_set_learning_rate(runner, seq_jsonl, tmp_path):
    d, _ = _checked(runner, seq_jsonl, tmp_path)
    r = runner.invoke(app, ["set", "learning-rate", "001", "scan"])
    assert r.exit_code == 0, r.stdout
    assert _out(r) == {"learning_rate": "scan"}
    assert _state_of(tmp_path).lr_choice == "scan"
    assert runner.invoke(app, ["set", "learning-rate", "001", "default"]).exit_code == 0
    assert _state_of(tmp_path).lr_choice == "default"
    r = runner.invoke(app, ["set", "learning-rate", "001", "huge"])
    assert r.exit_code == 2 and "default or scan" in _out(r)["error"]
    store.new_run(d, "noise", {}, [0, 1, 2], "noise floor")
    r = runner.invoke(app, ["set", "learning-rate", "001", "scan"])
    assert r.exit_code == 2
    assert _out(r)["error"] == "the learning rate is fixed once the experiment has started"
    assert _state_of(tmp_path).lr_choice == "default"


def test_set_learning_rate_needs_checked_data(runner, seq_jsonl, tmp_path):
    _new(runner, seq_jsonl)
    r = runner.invoke(app, ["set", "learning-rate", "001", "scan"])
    assert r.exit_code == 2 and _out(r)["error"] == "check the data first (step: check-data)"
    _draft(runner, "other")
    r = runner.invoke(app, ["set", "learning-rate", "002", "scan"])
    assert r.exit_code == 2 and "choose the data first" in _out(r)["error"]


def test_set_experiments_writes_budget(runner, seq_jsonl, tmp_path):
    _new(runner, seq_jsonl)
    r = runner.invoke(app, [
        "set", "experiments", "001", "--max-runs", "12", "--max-hours", "3.5", "--seeds", "4", "--subset", "50",
    ])
    assert r.exit_code == 0, r.stdout
    assert _out(r) == {"max_runs": 12, "max_hours": 3.5, "seeds": 4, "subset": 50}
    cfg = _cfg_of(tmp_path)
    assert (cfg.budget.max_runs, cfg.budget.max_hours, cfg.seeds, cfg.research_subset) == (12, 3.5, 4, 50)
    assert _state_of(tmp_path).budget_chosen is True


def test_set_experiments_keeps_unspecified_values(runner, seq_jsonl, tmp_path):
    _new(runner, seq_jsonl, "--max-runs", "9")
    r = runner.invoke(app, ["set", "experiments", "001", "--seeds", "5"])
    assert r.exit_code == 0, r.stdout
    cfg = _cfg_of(tmp_path)
    assert cfg.seeds == 5 and cfg.budget.max_runs == 9


@pytest.mark.parametrize("args,msg", [
    (["--seeds", "1"], "at least 2"),
    (["--max-runs", "0"], "max-runs"),
    (["--max-hours", "0"], "max-hours"),
    (["--subset", "0"], "subset"),
])
def test_set_experiments_validation(runner, seq_jsonl, tmp_path, args, msg):
    _new(runner, seq_jsonl)
    before = (_exp_dir(tmp_path) / "experiment.yaml").read_text()
    r = runner.invoke(app, ["set", "experiments", "001", *args])
    assert r.exit_code == 2 and msg in _out(r)["error"]
    assert (_exp_dir(tmp_path) / "experiment.yaml").read_text() == before
    assert _state_of(tmp_path).budget_chosen is False


def test_set_experiments_refused_on_draft_and_after_noise(runner, seq_jsonl, tmp_path):
    _draft(runner)
    r = runner.invoke(app, ["set", "experiments", "001", "--seeds", "3"])
    assert r.exit_code == 2 and "choose the data first" in _out(r)["error"]
    _new(runner, seq_jsonl, name="full")
    d = tmp_path / "experiments" / "002-full"
    store.new_run(d, "noise", {}, [0, 1, 2], "noise floor")
    r = runner.invoke(app, ["set", "experiments", "002", "--seeds", "3"])
    assert r.exit_code == 2
    assert _out(r)["error"] == "the experiment budget is fixed once the experiment has started"


def test_save_config_normal_mode_and_cleanup(tmp_path, monkeypatch):
    import os
    import stat

    cfg = ex.ExperimentConfig(name="x", task="seq-cls")
    d = tmp_path / "e"
    d.mkdir()
    old = os.umask(0o022)
    try:
        ex.save_config(d, cfg)
        assert stat.S_IMODE((d / "experiment.yaml").stat().st_mode) == 0o644
    finally:
        os.umask(old)

    def boom(*a, **k):
        raise RuntimeError("dump failed")

    monkeypatch.setattr(ex.yaml, "safe_dump", boom)
    with pytest.raises(RuntimeError):
        ex.save_config(d, cfg)
    assert sorted(p.name for p in d.iterdir()) == ["experiment.yaml"]


def test_baseline_checksum_locks_budget_and_lr_before_noise_request(runner, seq_jsonl, tmp_path):
    d, _ = _checked(runner, seq_jsonl, tmp_path)
    ex.record_baseline_checksums(d, store.profile_for(ex.load(d)[0], {}, store.project_root(d)))
    assert not any(r["stage"] == "noise" for r in store.list_requests(d))
    before = (d / "experiment.yaml").read_text()
    r = runner.invoke(app, ["set", "experiments", "001", "--seeds", "5"])
    assert r.exit_code == 2 and "fixed once the experiment has started" in _out(r)["error"]
    r = runner.invoke(app, ["set", "learning-rate", "001", "scan"])
    assert r.exit_code == 2 and "fixed once the experiment has started" in _out(r)["error"]
    assert (d / "experiment.yaml").read_text() == before
    ex.verify_checksums(d, store.project_root(d))


@pytest.mark.parametrize("args,msg", [(["--max-runs", "0"], "max-runs"), (["--max-hours", "0"], "max-hours")])
def test_new_refuses_empty_budget(runner, seq_jsonl, tmp_path, args, msg):
    r = _new(runner, seq_jsonl, *args)
    assert r.exit_code == 2 and msg in _out(r)["error"]
    assert not (tmp_path / "experiments").exists()


def test_set_losses_after_noise_opens_next_round(runner, seq_jsonl, tmp_path):
    d, _ = _checked(runner, seq_jsonl, tmp_path)
    with ex.state_lock(d):
        st = ex.load(d)[2]
        st.sigma_obj = 0.01
        ex.save_state(d, st)
    assert _state_of(tmp_path).round_open is False
    assert runner.invoke(app, ["set", "losses", "001", "ce,focal"]).exit_code == 0
    assert _state_of(tmp_path).round_open is True


# ---- next / steps follow the guided order ----


def _statuses(runner, *args):
    r = runner.invoke(app, ["steps", *args])
    assert r.exit_code == 0, r.stdout
    out = _out(r)
    return out, {x["step"]: x["status"] for x in out["steps"]}


def test_next_full_walk_fresh_experiment(runner, seq_jsonl, tmp_path):
    _draft(runner)
    seq_jsonl(_seq_rows())
    d = _exp_dir(tmp_path)
    assert _next(runner)["stage"] == "choose-data"
    assert runner.invoke(app, ["set", "data", "001", "--train", "data.jsonl"]).exit_code == 0
    assert _next(runner)["stage"] == "choose-model"
    assert runner.invoke(app, ["set", "model", "001", "tiny-bert"]).exit_code == 0
    assert _next(runner)["stage"] == "choose-goal"
    assert runner.invoke(app, ["set", "goal", "001", "--metric", "macro_f1"]).exit_code == 0
    assert _next(runner)["stage"] == "check-data"
    assert runner.invoke(app, ["check-data", "001"]).exit_code == 0
    assert _next(runner)["stage"] == "choose-loss"
    assert runner.invoke(app, ["set", "losses", "001", "focal"]).exit_code == 0
    assert _next(runner)["stage"] == "choose-hyperparameters"
    _space(d)
    out = _next(runner)
    assert out["stage"] == "choose-learning-rate" and out["reason"] == "learning speed not chosen yet"
    assert runner.invoke(app, ["set", "learning-rate", "001", "scan"]).exit_code == 0
    out = _next(runner)
    assert out["stage"] == "choose-learning-rate" and out["reason"] == "the learning-speed test hasn't finished"
    _set_state(d, lr_scan_best=1e-4)
    assert _next(runner)["stage"] == "choose-experiments"
    _set_state(d, budget_chosen=True)
    out = _next(runner)
    assert out["stage"] == "start-experiment" and out["reason"] == "the starting-point measurement hasn't run yet"
    store.new_run(d, "noise", {}, [0, 1, 2], "noise floor")  # queued, not finished
    out = _next(runner)
    assert out["stage"] == "start-experiment" and out["reason"] == "the starting-point measurement is running"
    _set_state(d, sigma_obj=0.01, round=1)
    out = _next(runner)
    assert out["stage"] == "start-experiment" and out["alternatives"][0]["stage"] == "status"


def test_next_round_two_skips_choose_loss_when_round_open(runner, seq_jsonl, tmp_path):
    assert _new(runner, seq_jsonl).exit_code == 0
    d = _exp_dir(tmp_path)
    _splits(d)
    _space(d)
    _set_state(d, sigma_obj=0.01, round=1)
    (d / "log.md").write_text("## Summary\n")
    (d / "report.md").write_text("r")
    assert _next(runner)["stage"] == "finish"
    assert runner.invoke(app, ["set", "losses", "001", "focal"]).exit_code == 0
    assert _state_of(tmp_path).round_open is True
    assert _next(runner)["stage"] == "choose-hyperparameters"
    _set_state(d, round_open=False)
    assert _next(runner)["stage"] == "finish"


def test_next_old_one_shot_experiment_after_noise(runner, seq_jsonl, tmp_path):
    assert _new(runner, seq_jsonl).exit_code == 0
    d = _exp_dir(tmp_path)
    _splits(d)
    _space(d)
    _set_state(d, sigma_obj=0.01, round=1)  # no lr_choice, no budget_chosen
    out = _next(runner)
    assert out["stage"] == "start-experiment" and "still running" in out["reason"]


def test_steps_statuses_at_three_points(runner, seq_jsonl, tmp_path):
    out, st = _statuses(runner)
    assert out["experiment"] is None and out["next"]["stage"] == "choose-task"
    assert st["start"] == "to do" and st["choose-task"] == "next" and st["status"] == "optional"
    assert st["compare-models"] == "optional" and st["finish"] == "to do"
    assert [x.get("number") for x in out["steps"]][:3] == [None, 1, 2] and out["steps"][-1]["number"] == 14

    _draft(runner)
    seq_jsonl(_seq_rows())
    assert runner.invoke(app, ["set", "data", "001", "--train", "data.jsonl"]).exit_code == 0
    assert runner.invoke(app, ["set", "model", "001", "tiny-bert"]).exit_code == 0
    out, st = _statuses(runner, "001")
    assert st["start"] == "done" and st["choose-task"] == "done" and st["choose-data"] == "done"
    assert st["choose-goal"] == "next" and st["check-data"] == "to do"
    assert st["compare-models"] == "skipped"
    assert runner.invoke(app, ["set", "goal", "001", "--metric", "macro_f1"]).exit_code == 0

    d = _exp_dir(tmp_path)
    _splits(d)
    _space(d)
    _set_state(d, sigma_obj=0.01, round=1, promoted=True)
    (d / "log.md").write_text("## Summary\n")
    (d / "report.md").write_text("r")
    out, st = _statuses(runner)
    assert out["next"]["stage"] == "choose-task"
    assert all(v == "done" for k, v in st.items() if k not in ("status", "compare-models"))
    assert st["status"] == "optional"


def test_steps_round_two_keeps_once_only_steps_done(runner, seq_jsonl, tmp_path):
    assert _new(runner, seq_jsonl).exit_code == 0
    d = _exp_dir(tmp_path)
    _splits(d)
    _space(d)
    _set_state(d, sigma_obj=0.01, round=1, lr_choice="default", budget_chosen=True)
    (d / "log.md").write_text("## Summary\n")
    (d / "report.md").write_text("r")
    assert runner.invoke(app, ["set", "losses", "001", "focal"]).exit_code == 0
    out, st = _statuses(runner)
    assert out["next"]["stage"] == "choose-hyperparameters"
    assert st["choose-loss"] == "done" and st["choose-hyperparameters"] == "next"
    assert st["choose-learning-rate"] == "done" and st["choose-experiments"] == "done"
    assert st["start-experiment"] == "to do" and st["report"] == "to do" and st["finish"] == "to do"
    assert st["check-data"] == "done" and st["choose-goal"] == "done"
