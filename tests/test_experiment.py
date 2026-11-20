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


def test_commit_space_new_round_resets_non_keep_streak(tmp_path):
    _, d = _setup(tmp_path)
    space = ex.SearchSpace(round=0, tunable=["lr"], losses=["ce"])
    ex.commit_space(d, space)
    state = ex.load(d)[2]
    state.non_keep_streak = 7
    ex.save_state(d, state)
    assert ex.commit_space(d, space).non_keep_streak == 0
    assert ex.load(d)[2].non_keep_streak == 0


def test_draft_config_and_missing_setup(tmp_path):
    cfg = ex.ExperimentConfig(name="d", task="seq-cls")
    assert cfg.model is None and cfg.data is None and cfg.objective is None
    assert ex.missing_setup(cfg) == "choose-data"
    cfg.data = ex.DataPaths(train="t.jsonl")
    assert ex.missing_setup(cfg) == "choose-model"
    cfg.model = "tiny-bert"
    assert ex.missing_setup(cfg) == "choose-goal"
    cfg.objective = ex.MetricRef(metric="macro_f1")
    assert ex.missing_setup(cfg) is None
    ex.require_setup(cfg)


@pytest.mark.parametrize("field,msg", [
    ("data", "choose the data first (step: choose-data)"),
    ("model", "choose the model first (step: choose-model)"),
    ("objective", "choose the goal first (step: choose-goal)"),
])
def test_require_setup_messages(field, msg):
    cfg = _cfg()
    setattr(cfg, field, None)
    with pytest.raises(ValueError, match=r"^" + msg.replace("(", r"\(").replace(")", r"\)") + "$"):
        ex.require_setup(cfg)


def test_draft_roundtrips_and_old_yaml_loads(tmp_path):
    d = ex.new_experiment(tmp_path, "draft", ex.ExperimentConfig(name="draft", task="token-cls"))
    cfg, _, _ = ex.load(d)
    assert cfg.model is None and cfg.task == "token-cls"
    old = ex.new_experiment(tmp_path, "old", _cfg())
    assert ex.load(old)[0].objective.metric == "f1_macro"
