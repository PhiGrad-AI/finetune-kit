import math
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError
from torch import nn
from torch.nn import functional as F

from finetune_kit.catalog import ENTRIES, validate_config
from finetune_kit.profiles import load_profiles
from finetune_kit.train.config import TrainConfig, build_config, defaults_for, nondefault_count
from finetune_kit.train.losses import class_weights, seq_loss, token_loss
from finetune_kit.train.models import (
    SeqClassifier, TokenClassifier, apply_lora, collate, encode_seq, encode_tokens, load_backbone,
)
from finetune_kit.train.optim import lr_factor, param_groups


@pytest.fixture
def prof(monkeypatch):
    monkeypatch.setenv("FINETUNE_KIT_TEST_PROFILES", "1")
    return load_profiles(None)["tiny-bert"]


@pytest.fixture
def cfg(prof):
    def make(task="seq-cls", **kw):
        return build_config(kw, prof, task, 0)

    return make


def test_every_catalog_key_is_a_field():
    assert set(ENTRIES) <= set(TrainConfig.model_fields)


def test_defaults_come_from_profile_and_catalog(prof, cfg):
    d = defaults_for(prof, "seq-cls")
    assert d["lr"] == prof.lr_default and d["max_len"] == prof.max_len and d["batch"] == prof.batch
    assert "crf" not in d and "pooling" in d
    c = cfg(lr=2e-3, class_weight="balanced")
    assert c.lr == 2e-3 and c.class_weight == "balanced" and c.micro == prof.micro
    assert c.model == "tiny-bert" and c.beta1 == 0.9 and c.warmup == 0.05
    assert nondefault_count({"lr": prof.lr_default, "dropout": 0.2, "ema": 0.99}, d) == 2


def test_unknown_key_rejected(prof):
    with pytest.raises(ValidationError):
        build_config({"nope": 1}, prof, "seq-cls", 0)


def test_dice_with_crf_rejected():
    d = {"loss": "ce", "crf": False}
    errs = validate_config({"loss": "dice", "crf": True}, ["loss", "crf"], ["ce", "dice"], d, "token-cls")
    assert any("crf" in e for e in errs)
    assert validate_config({"loss": "dice"}, ["loss"], ["ce", "dice"], d, "token-cls") == []


class _Stub(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.config = SimpleNamespace(num_hidden_layers=2)
        self.backbone.embeddings = nn.Linear(2, 2)
        self.backbone.encoder = nn.Module()
        self.backbone.encoder.layer = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self.head = nn.Linear(2, 2)


def test_llrd_multipliers(cfg):
    groups = param_groups(_Stub(), cfg(llrd=0.5, head_lr_mult=10.0))
    mults = {g["base_mult"] for g in groups}
    assert {1.0, 0.5, 0.25} <= mults and 10.0 in mults
    assert {g["weight_decay"] for g in groups} == {0.0, 0.01}
    g = next(g for g in groups if g["base_mult"] == 0.25)
    assert g["lr"] == pytest.approx(cfg().lr * 0.25)


def test_lr_factor_warmup_and_end(cfg):
    c = cfg()
    assert lr_factor(0, 100, c) == pytest.approx(0.2)
    assert lr_factor(4, 100, c) == pytest.approx(1.0)
    assert lr_factor(99, 100, c) < 0.02
    cos = cfg(schedule="cosine", warmup=0.0)
    assert lr_factor(50, 100, cos) == pytest.approx(0.5, abs=0.02)
    assert lr_factor(7, 100, cfg(schedule="constant")) == 1.0


def _logits_y():
    g = torch.Generator().manual_seed(0)
    return torch.randn(6, 2, generator=g), torch.tensor([0, 1, 1, 0, 1, 0])


def test_focal_gamma0_equals_ce(cfg):
    x, y = _logits_y()
    f = seq_loss(x, y, cfg(loss="focal", focal_gamma=0.0), None, 1)
    assert f.item() == pytest.approx(F.cross_entropy(x, y, reduction="sum").item())


def test_ce_rank_weight0_equals_ce(cfg):
    x, y = _logits_y()
    a = seq_loss(x, y, cfg(loss="ce+rank", rank_weight=0.0), None, 1)
    assert a.item() == pytest.approx(F.cross_entropy(x, y, reduction="sum").item())
    assert seq_loss(x, y, cfg(loss="rank"), None, 1).item() > 0


def test_rank_needs_two_classes(cfg):
    with pytest.raises(ValueError, match="2 classes"):
        seq_loss(torch.randn(4, 3), torch.tensor([0, 1, 2, 0]), cfg(loss="rank"), None, 1)


def test_rank_uses_pos_margin(cfg):
    x = torch.tensor([[3.0, -3.0], [-3.0, 3.0]])
    y = torch.tensor([0, 1])
    good = seq_loss(x, y, cfg(loss="rank"), None, 1)
    flipped = seq_loss(x, y, cfg(loss="rank"), None, 0)  # pos=0: sample 0 is the positive
    assert good.item() == pytest.approx(flipped.item())
    assert good.item() < 0.01


def test_asl_zero_gammas_zero_clip_equals_bce(cfg):
    x, y = _logits_y()
    c = cfg(loss="asl", asl_gamma_neg=0.0, asl_gamma_pos=0.0, asl_clip=0.0)
    ref = F.binary_cross_entropy_with_logits(x, F.one_hot(y, 2).float(), reduction="sum")
    assert seq_loss(x, y, c, None, 1).item() == pytest.approx(ref.item(), rel=1e-5)


def test_asl_clip_ignores_easy_negatives(cfg):
    logit = math.log(0.03 / 0.97)  # p = 0.03 for the negative class
    x = torch.tensor([[logit, 0.0]])  # target class 1; class 0 is a negative with p=0.03
    y = torch.tensor([1])
    with_clip = seq_loss(x, y, cfg(loss="asl", asl_clip=0.05, asl_gamma_neg=0.0), None, 1)
    only_pos = -torch.log(torch.sigmoid(torch.tensor(0.0)))  # gamma_pos=0
    assert with_clip.item() == pytest.approx(only_pos.item(), rel=1e-5)
    no_clip = seq_loss(x, y, cfg(loss="asl", asl_clip=0.0, asl_gamma_neg=0.0), None, 1)
    assert no_clip.item() > with_clip.item()


def test_class_weight_balanced_inverse_freq(cfg):
    w = class_weights(cfg(class_weight="balanced"), [0, 0, 0, 1], 2, 1)
    assert w.tolist() == pytest.approx([4 / (2 * 3), 4 / (2 * 1)])
    assert class_weights(cfg(), [0, 1], 2, 1) is None
    assert class_weights(cfg(class_weight=5.0), [0, 1], 2, 1).tolist() == [1.0, 5.0]
    x, y = _logits_y()
    w = class_weights(cfg(class_weight=5.0), y.tolist(), 2, 1)
    weighted = seq_loss(x, y, cfg(class_weight=5.0), w, 1)
    assert weighted.item() > seq_loss(x, y, cfg(), None, 1).item()


def _tok_batch():
    g = torch.Generator().manual_seed(1)
    return torch.randn(2, 5, 3, generator=g), torch.tensor([[-100, 0, 1, 2, -100], [-100, 0, 0, -100, -100]])


def test_dice_weight1_equals_ce(cfg):
    x, y = _tok_batch()
    d = token_loss(x, y, cfg("token-cls", loss="dice", dice_ce_weight=1.0), None, 0)
    ce = F.cross_entropy(x.reshape(-1, 3), y.reshape(-1), ignore_index=-100, reduction="sum")
    assert d.item() == pytest.approx(ce.item())
    assert token_loss(x, y, cfg("token-cls"), None, 0).item() == pytest.approx(ce.item())


def test_dice_perfect_prediction_near_zero(cfg):
    y = torch.tensor([[-100, 0, 1, 2, 1, -100]])
    x = torch.full((1, 6, 3), -20.0)
    for i, t in enumerate(y[0].tolist()):
        if t >= 0:
            x[0, i, t] = 20.0
    c = cfg("token-cls", loss="dice", dice_ce_weight=0.0, dice_smooth=1.0)
    assert token_loss(x, y, c, None, 0).item() < 1e-3


def test_dice_ignores_padding(cfg):
    x, y = _tok_batch()
    c = cfg("token-cls", loss="dice", dice_ce_weight=0.5)
    base = token_loss(x, y, c, None, 0)
    x2 = torch.cat([x, torch.randn(2, 3, 3)], 1)
    y2 = torch.cat([y, torch.full((2, 3), -100)], 1)
    assert token_loss(x2, y2, c, None, 0).item() == pytest.approx(base.item())


def test_focal_token_gamma0_equals_ce(cfg):
    x, y = _tok_batch()
    ce = token_loss(x, y, cfg("token-cls"), None, 0)
    f = token_loss(x, y, cfg("token-cls", loss="focal", focal_gamma=0.0), None, 0)
    assert f.item() == pytest.approx(ce.item())


# ---- model-dependent tests -------------------------------------------------

def _batch(prof_tok, n=3):
    ids, mask = collate(encode_seq(prof_tok, ["hello world"] * n, 16)[0], prof_tok.pad_token_id, "cpu")
    return ids, mask


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")


@pytest.mark.model
def test_backbone_dropout_applied(prof):
    bb = load_backbone(prof, "cpu", 0.25, 0.15)
    assert bb.config.hidden_dropout_prob == 0.25 and bb.config.attention_probs_dropout_prob == 0.15
    assert next(bb.parameters()).dtype == torch.float32


@pytest.mark.model
def test_mean_pooling_ignores_padding(prof, cfg, tok):
    m = SeqClassifier(load_backbone(prof, "cpu", None, None), 2, cfg(pooling="mean")).eval()
    ids = torch.tensor([[2, 5, 6, 7, 3]])
    out = m(ids, torch.ones_like(ids))
    padded = torch.cat([ids, torch.zeros(1, 4, dtype=torch.long)], 1)
    mask = torch.cat([torch.ones_like(ids), torch.zeros(1, 4, dtype=torch.long)], 1)
    assert torch.allclose(out, m(padded, mask), atol=1e-5)


@pytest.mark.model
@pytest.mark.parametrize("pooling", ["cls", "attention", "weighted_layers"])
def test_pooling_shapes(prof, cfg, tok, pooling):
    c = cfg(pooling=pooling, pool_layers=3)
    m = SeqClassifier(load_backbone(prof, "cpu", None, None), 3, c)
    ids, mask = _batch(tok)
    assert m(ids, mask).shape == (3, 3)
    if pooling == "weighted_layers":
        assert m.pool.layer_weights.numel() == 3
        assert any(n.startswith("pool.") for n, _ in m.named_parameters())


@pytest.mark.model
def test_weighted_layers_too_many_layers(prof, cfg):
    with pytest.raises(ValueError, match="pool_layers"):
        SeqClassifier(load_backbone(prof, "cpu", None, None), 2, cfg(pooling="weighted_layers", pool_layers=12))


@pytest.mark.model
def test_msd_eval_is_deterministic(prof, cfg, tok):
    m = SeqClassifier(load_backbone(prof, "cpu", None, None), 2, cfg(msd_samples=4, dropout=0.5, head="mlp"))
    ids, mask = _batch(tok)
    m.eval()
    assert torch.equal(m(ids, mask), m(ids, mask))
    m.train()
    assert not torch.equal(m(ids, mask), m(ids, mask))


@pytest.mark.model
@pytest.mark.filterwarnings("ignore:.*is_gradient_checkpointing:FutureWarning")
def test_lora_trainable_params_small(prof, cfg):
    # The tiny model has hidden size 32, so rank 8 is ~18% of it; rank 1 checks the <5% property.
    bb = load_backbone(prof, "cpu", None, None)
    total = sum(p.numel() for p in bb.parameters())
    lora = apply_lora(bb, prof, 1, 16)
    c = cfg(lora_r=1)
    m = SeqClassifier(lora, 2, c)
    backbone_trainable = sum(p.numel() for p in m.backbone.parameters() if p.requires_grad)
    assert 0 < backbone_trainable < 0.05 * total
    assert all(p.requires_grad for p in m.head.parameters())
    groups = param_groups(m, cfg(llrd=0.5, lora_r=1))
    assert {g["base_mult"] for g in groups} >= {1.0, 0.5}
    assert sum(p.numel() for g in groups for p in g["params"]) == backbone_trainable + sum(
        p.numel() for p in list(m.head.parameters()) + list(m.pool.parameters())
    )


@pytest.mark.model
def test_crf_nll_and_decode(prof, cfg, tok):
    m = TokenClassifier(load_backbone(prof, "cpu", None, None), 3, cfg("token-cls", crf=True))
    ids, mask = _batch(tok, 2)
    logits = m(ids, mask)
    labels = torch.full(ids.shape, -100)
    labels[0, 1:3] = torch.tensor([1, 2])
    labels[1, 1] = 0
    nll = m.nll(logits, labels)
    assert nll.ndim == 0 and nll.item() > 0
    nll.backward()
    assert m.crf.transitions.grad is not None
    paths = m.decode(logits, labels != -100)
    assert [len(p) for p in paths] == [2, 1]
    assert m.nll(logits, torch.full(ids.shape, -100)).item() == 0


@pytest.mark.model
def test_token_truncation_keeps_alignment(tok):
    t2i = {"O": 0, "B-PER": 1, "I-PER": 2}
    ids, labels, words, n_trunc = encode_tokens(
        tok, [["John"] + ["x"] * 600], [["B-PER"] + ["O"] * 600], t2i, 64, "first"
    )
    assert n_trunc == 1 and len(ids[0]) == 64 and len(labels[0]) == 64 == len(words[0])
    assert labels[0][1] == t2i["B-PER"] and labels[0][0] == -100 and labels[0][-1] == -100
    ids, n_trunc = encode_seq(tok, ["word " * 300, "short"], 32)
    assert n_trunc == 1 and len(ids[0]) == 32 and len(ids[1]) < 32


@pytest.mark.model
def test_align_all_converts_B_to_I(tok):
    t2i = {"O": 0, "B-PER": 1, "I-PER": 2}
    word = "internationalization"
    n_sub = len(tok.tokenize(word))
    assert n_sub > 1
    _, first, _, _ = encode_tokens(tok, [[word]], [["B-PER"]], t2i, 64, "first")
    _, allv, _, _ = encode_tokens(tok, [[word]], [["B-PER"]], t2i, 64, "all")
    assert first[0] == [-100, 1] + [-100] * (n_sub - 1) + [-100]
    assert allv[0] == [-100, 1] + [2] * (n_sub - 1) + [-100]


def test_collate_pads():
    ids, mask, y = collate([[1, 2, 3], [4]], 0, "cpu", labels=[[9, 9, 9], [8]])
    assert ids.tolist() == [[1, 2, 3], [4, 0, 0]] and mask.sum().item() == 4
    assert y.tolist() == [[9, 9, 9], [8, -100, -100]]
