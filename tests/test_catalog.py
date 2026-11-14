import yaml

from finetune_kit.catalog import ENTRIES, listing, names, recommend, validate_config
from finetune_kit.profiles import ModelProfile, add_profile, load_profiles

DEFAULTS = {
    "lr": 3e-5, "warmup": 0.05, "loss": "ce", "lora_r": 0, "lora_alpha": 16,
    "hidden_dropout": None, "attn_dropout": None,
}
SEQ_LOSSES = ["ce", "focal", "asl", "rank", "ce+rank"]
BALANCED = {"minority_share": 0.5, "token_len": {"p95": 400}}


def test_profiles_load_and_pins():
    ps = load_profiles(None)
    p = ps["ultrabert-150m"]
    assert p.trust_remote_code and p.lr_default == 8e-5
    assert p.revision == "545ac9d9c7aece444f014ada5a1e9818132b4d35"
    assert p.dropout_keys == {"hidden": [], "attn": ["attention_dropout"]}
    assert "o_proj" in p.lora_targets
    d = ps["deberta-v3-base"]
    assert d.force_fp32_on_mps and d.grad_checkpointing and d.lora_targets[0] == "query_proj"
    assert ps["modernbert-base"].lora_targets == ["Wqkv", "Wo", "Wi"]
    assert ps["roberta-large"].micro == 2 and ps["roberta-base"].micro == 4
    assert all(len(p.revision) == 40 for p in ps.values())
    assert "tiny-bert" not in ps


def test_test_profiles_env(monkeypatch):
    monkeypatch.setenv("FINETUNE_KIT_TEST_PROFILES", "1")
    ps = load_profiles(None)
    assert ps["tiny-bert"].hf_id == "hf-internal-testing/tiny-random-bert"
    assert ps["tiny-roberta"].max_len == 64


def test_project_override(tmp_path):
    (tmp_path / ".finetune-kit").mkdir()
    (tmp_path / ".finetune-kit" / "models.yaml").write_text(yaml.safe_dump({
        "roberta-base": {**_as_dict(load_profiles(None)["roberta-base"]), "lr_default": 7e-5},
    }))
    assert load_profiles(tmp_path)["roberta-base"].lr_default == 7e-5
    assert load_profiles(None)["roberta-base"].lr_default == 2e-5


def _as_dict(p):
    d = dict(p.__dict__)
    d.pop("key")
    d["lr_range"] = list(d["lr_range"])
    return d


def test_add_profile_generic(tmp_path):
    cfg = {"model_type": "bert", "max_position_embeddings": 1024,
           "hidden_dropout_prob": 0.1, "attention_probs_dropout_prob": 0.1}
    p = add_profile(tmp_path, "acme/My_Model", False, config=cfg)
    assert p.max_len == 512 and p.notes == "generic: lr-scan required" and p.lr_default == 3e-5
    assert p.dropout_keys["hidden"] == ["hidden_dropout_prob"]
    again = load_profiles(tmp_path)[p.key]
    assert again == p and again.hf_id == "acme/My_Model"
    q = add_profile(tmp_path, "acme/other", True, config={"model_type": "x", "max_position_embeddings": 128})
    assert q.max_len == 128 and q.dropout_keys == {} and q.lora_targets == [] and q.trust_remote_code
    assert {p.key, q.key} <= set(load_profiles(tmp_path))


def test_recommend_imbalance():
    r = recommend({"minority_share": 0.18, "token_len": {"p95": 400}}, "f1", "seq-cls")
    assert {"lr", "llrd", "class_weight", "loss:focal"} <= set(r)
    assert "max_len" not in r


def test_recommend_max_len_short():
    assert "max_len" in recommend({"token_len": {"p95": 100}}, "f1", "seq-cls")


def test_recommend_auc_metric_rank():
    assert "loss:ce+rank" in recommend(BALANCED, "pauc_hi", "seq-cls")


def test_recommend_asl_recall_first():
    assert "loss:asl" in recommend(BALANCED, "skip_at_recall", "seq-cls")
    assert "loss:asl" not in recommend(BALANCED, "skip_at_recall", "token-cls")


def test_recommend_dice_rare_entities():
    r = recommend({"entity_token_share": 0.04, "token_len": {"p95": 400}}, "entity_f1", "token-cls")
    assert "loss:dice" in r


def test_listing_defaults_from_profile():
    p = load_profiles(None)["roberta-base"]
    rows = {r["name"]: r for r in listing("seq-cls", p, {"minority_share": 0.1}, "f1")}
    assert rows["lr"]["default"] == 2e-5 and rows["lr"]["recommended"]
    assert rows["max_len"]["default"] == 512
    assert rows["class_weight"]["recommended"] and "focal" in rows["loss"]["reason"]
    assert "pooling" in rows and "crf" not in rows


def test_dice_not_for_seq():
    errs = validate_config({"loss": "dice"}, ["loss"], SEQ_LOSSES, DEFAULTS, "seq-cls")
    assert errs


def test_validate_locked_key():
    errs = validate_config({"warmup": 0.1}, ["lr"], ["ce"], DEFAULTS, "seq-cls")
    assert any("warmup" in e and "not tunable" in e for e in errs)


def test_validate_default_value_ok():
    assert validate_config({"warmup": 0.05}, ["lr"], ["ce"], DEFAULTS, "seq-cls") == []


def test_validate_unknown_key():
    errs = validate_config({"bogus": 1}, ["bogus"], ["ce"], DEFAULTS, "seq-cls")
    assert any("unknown" in e for e in errs)
    errs = validate_config({"crf": True}, ["crf"], ["ce"], DEFAULTS, "seq-cls")
    assert any("unknown" in e for e in errs)


def test_validate_string_number():
    errs = validate_config({"lr": "4e-5"}, ["lr"], ["ce"], DEFAULTS, "seq-cls")
    assert any("lr" in e and "number" in e for e in errs)


def test_validate_hard_limit():
    errs = validate_config({"lr": 0.5}, ["lr"], ["ce"], DEFAULTS, "seq-cls")
    assert any("1e-3" in e for e in errs)
    assert validate_config({"lr": 4e-5}, ["lr"], ["ce"], DEFAULTS, "seq-cls") == []


def test_validate_max_len_uses_profile():
    p = load_profiles(None)["roberta-base"]
    assert validate_config({"max_len": 256}, ["max_len"], ["ce"], DEFAULTS, "seq-cls", p) == []
    assert validate_config({"max_len": 1024}, ["max_len"], ["ce"], DEFAULTS, "seq-cls", p)


def test_validate_loss_not_allowed():
    errs = validate_config({"loss": "rank"}, ["loss"], ["ce", "focal"], DEFAULTS, "seq-cls")
    assert any("loss" in e for e in errs)
    assert validate_config({"loss": "focal"}, ["loss"], ["ce", "focal"], DEFAULTS, "seq-cls") == []


def test_ema_zero_or_range():
    assert validate_config({"ema": 0}, ["ema"], ["ce"], {}, "seq-cls") == []
    assert validate_config({"ema": 0.99}, ["ema"], ["ce"], {}, "seq-cls") == []
    assert validate_config({"ema": 0.5}, ["ema"], ["ce"], {}, "seq-cls")


def test_token_task_has_no_pooling():
    assert "pooling" not in names("token-cls")
    assert "crf" in names("token-cls") and "crf" not in names("seq-cls")
    assert "dice_smooth" in ENTRIES


def test_backbone_dropout_rejected_without_keys(tmp_path):
    g = add_profile(tmp_path, "acme/odd", False, config={"model_type": "x"})
    errs = validate_config({"hidden_dropout": 0.2}, ["hidden_dropout"], ["ce"], DEFAULTS, "seq-cls", g)
    assert any("not supported for this model" in e for e in errs)
    errs = validate_config({"lora_r": 8}, ["lora_r"], ["ce"], DEFAULTS, "seq-cls", g)
    assert any("not supported for this model" in e for e in errs)
    ok = load_profiles(None)["roberta-base"]
    assert validate_config({"hidden_dropout": 0.2}, ["hidden_dropout"], ["ce"], DEFAULTS, "seq-cls", ok) == []
    ub = load_profiles(None)["ultrabert-150m"]
    assert validate_config({"hidden_dropout": 0.2}, ["hidden_dropout"], ["ce"], DEFAULTS, "seq-cls", ub)


def test_lora_alpha_requires_lora_r():
    errs = validate_config({"lora_alpha": 32}, ["lora_alpha"], ["ce"], DEFAULTS, "seq-cls")
    assert any("lora_alpha" in e and "lora_r" in e for e in errs)
    assert validate_config({"lora_alpha": 32, "lora_r": 8}, ["lora_alpha", "lora_r"], ["ce"], DEFAULTS, "seq-cls") == []


def test_catalog_defaults_match_trainer_defaults():
    d = {n: e.default for n, e in ENTRIES.items()}
    assert d["warmup"] == 0.05 and d["epochs"] == 2 and d["beta2"] == 0.999
    assert d["lora_alpha"] == 16 and d["ema"] == 0.0 and d["batch"] == 8
    assert d["asl_gamma_neg"] == 4 and d["asl_clip"] == 0.05 and d["dice_ce_weight"] == 0.5
    assert d["hidden_dropout"] is None and d["attn_dropout"] is None


def test_partial_project_override_merges(tmp_path):
    (tmp_path / ".finetune-kit").mkdir()
    (tmp_path / ".finetune-kit" / "models.yaml").write_text(yaml.safe_dump({"roberta-base": {"lr_default": 3e-5}}))
    p = load_profiles(tmp_path)["roberta-base"]
    shipped = load_profiles(None)["roberta-base"]
    assert p.lr_default == 3e-5 and p.hf_id == shipped.hf_id and p.revision == shipped.revision


def test_new_project_profile_missing_fields_named(tmp_path):
    import pytest

    (tmp_path / ".finetune-kit").mkdir()
    (tmp_path / ".finetune-kit" / "models.yaml").write_text(yaml.safe_dump({"mine": {"hf_id": "acme/mine"}}))
    with pytest.raises(ValueError, match=r"profile 'mine'.*missing.*'lr_default'.*'revision'"):
        load_profiles(tmp_path)


def test_add_profile_refuses_existing_key(tmp_path):
    import pytest

    cfg = {"model_type": "x", "max_position_embeddings": 128}
    with pytest.raises(ValueError, match="'roberta-base' already exists"):
        add_profile(tmp_path, "FacebookAI/roberta-base", False, config=cfg)  # shipped
    add_profile(tmp_path, "acme/mine", False, config=cfg)
    with pytest.raises(ValueError, match="'mine' already exists"):
        add_profile(tmp_path, "other/mine", False, config=cfg)  # project


def test_validate_config_rejects_non_finite_numbers():
    import math

    tunable = ["lr", "warmup", "msd_samples"]
    for key, val in (("lr", math.inf), ("warmup", math.nan), ("msd_samples", math.inf), ("lr", -math.inf)):
        errs = validate_config({key: val}, tunable, ["ce"], DEFAULTS, "seq-cls")
        assert errs and "finite" in errs[0], (key, val, errs)


def test_not_tunable_hint_is_agent_neutral():
    errs = validate_config({"warmup": 0.1}, ["lr"], ["ce"], DEFAULTS, "seq-cls")
    assert errs == ["warmup is not tunable (locked at 0.05); re-run the hparams stage to open it"]
