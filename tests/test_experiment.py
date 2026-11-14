import pytest

from finetune_kit import experiment as ex


def _cfg(**kw):
    base = dict(
        name="demo exp", task="seq-cls", model="tiny-bert",
        data=ex.DataPaths(train="train.jsonl"),
        objective=ex.MetricRef(metric="f1_macro"),
    )
    base.update(kw)
    return ex.ExperimentConfig(**base)


def _setup(tmp_path):
    root = tmp_path
    d = ex.new_experiment(root, "Demo Exp!", _cfg())
    return root, d


def test_new_experiment_numbering(tmp_path):
    d1 = ex.new_experiment(tmp_path, "Demo Exp!", _cfg())
    d2 = ex.new_experiment(tmp_path, "second  one", _cfg())
    assert d1.name == "001-demo-exp"
    assert d2.name == "002-second-one"
    assert (d1 / "experiment.yaml").is_file() and (d1 / "state.json").is_file()


def test_roundtrip_yaml(tmp_path):
    cfg = _cfg(guardrail=ex.MetricRef(metric="recall", params={"at": 0.9}), seeds=5)
    d = ex.new_experiment(tmp_path, "rt", cfg)
    cfg2, space, state = ex.load(d)
    assert cfg2 == cfg
    assert space is None
    assert state == ex.State()
    state.round = 3
    state.bakeoff = {"a": 1}
    ex.save_state(d, state)
    assert ex.load(d)[2] == state


def test_commit_space_increments_round(tmp_path):
    root, d = _setup(tmp_path)
    s = ex.commit_space(d, ex.SearchSpace(round=0, tunable=["lr"], losses=["ce"]))
    assert s.round == 1
    assert "search-space.yaml" in s.checksums
    _, space, state = ex.load(d)
    assert space.tunable == ["lr"] and state.round == 1
    assert ex.commit_space(d, ex.SearchSpace(round=0, tunable=["lr"], losses=["ce"])).round == 2


def test_commit_space_rejects_unknown(tmp_path):
    _, d = _setup(tmp_path)
    with pytest.raises(ValueError, match="nope"):
        ex.commit_space(d, ex.SearchSpace(round=0, tunable=["nope"], losses=["ce"]))
    with pytest.raises(ValueError, match="dice"):
        ex.commit_space(d, ex.SearchSpace(round=0, tunable=["lr"], losses=["dice"]))


def test_tamper_detected(tmp_path):
    root, d = _setup(tmp_path)
    ex.commit_space(d, ex.SearchSpace(round=0, tunable=["lr"], losses=["ce"]))
    (root / ".finetune-kit" / "memory").mkdir(parents=True)
    (root / ".finetune-kit" / "memory" / "research-rules.md").write_text("rules")
    ex.record_checksums(d, root)
    ex.verify_checksums(d, root)
    (d / "search-space.yaml").write_text("round: 1\ntunable: [lr, wd]\nlosses: [ce]\n")
    (root / ".finetune-kit" / "memory" / "research-rules.md").unlink()
    with pytest.raises(ex.TamperError) as e:
        ex.verify_checksums(d, root)
    assert "search-space.yaml" in str(e.value)
    assert ".finetune-kit/memory/research-rules.md" in str(e.value)


def test_commit_after_record_is_not_tamper(tmp_path):
    root, d = _setup(tmp_path)
    ex.commit_space(d, ex.SearchSpace(round=0, tunable=["lr"], losses=["ce"]))
    ex.record_checksums(d, root)
    ex.commit_space(d, ex.SearchSpace(round=0, tunable=["lr", "weight_decay"], losses=["ce"]))
    ex.verify_checksums(d, root)
