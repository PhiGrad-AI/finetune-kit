"""Training loop: one model, one seed, returns validation predictions of the best checkpoint."""
import contextlib
import math
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from finetune_kit.paths import test_env
from finetune_kit.train.config import TrainConfig
from finetune_kit.train.losses import class_weights, seq_loss, token_loss
from finetune_kit.train.models import (
    SeqClassifier, TokenClassifier, apply_lora, collate, encode_seq, encode_tokens, load_backbone,
)
from finetune_kit.train.optim import lr_factor, param_groups

_MOD_LAYER = re.compile(r"(?:^|\.)layers?\.(\d+)(?:\.|$)")
_EVAL_BATCH = 16


@dataclass
class TrainResult:
    val_out: np.ndarray | list[list[str]]
    best_epoch: float
    train_s: float
    peak_gb: float
    n_truncated: int
    steps: int
    model: nn.Module | None


def pick_device() -> str:
    forced = os.environ.get("FINETUNE_KIT_DEVICE")  # tests pin cpu
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def reinit_top(backbone: nn.Module, n_top: int) -> None:
    """Fresh weights for the top n_top encoder layers (Linear: N(0, init_range), norms: 1/0)."""
    n_layers = backbone.config.num_hidden_layers
    std = getattr(backbone.config, "initializer_range", 0.02)
    hit = 0
    for name, mod in backbone.named_modules():
        m = _MOD_LAYER.search(name)
        if not m or int(m.group(1)) < n_layers - n_top:
            continue
        if isinstance(mod, nn.Linear):
            mod.weight.normal_(0.0, std)
            if mod.bias is not None:
                mod.bias.zero_()
            hit += 1
        elif "norm" in type(mod).__name__.lower() and getattr(mod, "weight", None) is not None:
            mod.weight.fill_(1.0)
            if getattr(mod, "bias", None) is not None:
                mod.bias.zero_()
            hit += 1
    if hit == 0:
        raise RuntimeError("reinit_top matched no modules")


def epoch_order(label_ids: list[int], oversample: float, rng: random.Random) -> list[int]:
    """One epoch of row indices; minority-class rows repeated round(oversample) times, shuffled."""
    k = max(1, round(oversample))
    order = list(range(len(label_ids)))
    if k > 1:
        counts: dict[int, int] = {}
        for y in label_ids:
            counts[y] = counts.get(y, 0) + 1
        minority = min(counts, key=lambda c: (counts[c], c))
        order += [i for i, y in enumerate(label_ids) if y == minority] * (k - 1)
    rng.shuffle(order)
    return order


def _epoch_len(label_ids: list[int], oversample: float) -> int:
    return len(epoch_order(label_ids, oversample, random.Random(0)))


def runtime_config(cfg: TrainConfig, profile, device: str) -> TrainConfig:
    """Profiles with force_fp32_on_mps run without bf16 autocast on MPS."""
    if cfg.precision == "bf16" and device == "mps" and profile.force_fp32_on_mps:
        return cfg.model_copy(update={"precision": "fp32"})
    return cfg


def _autocast(device: str, cfg: TrainConfig):
    if cfg.precision == "bf16":
        return torch.autocast(device_type=device, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _peak_gb(device: str) -> float:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device == "mps":
        return torch.mps.current_allocated_memory() / 1e9
    return 0.0


def _sym_kl(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    la, lb = F.log_softmax(a.float(), -1), F.log_softmax(b.float(), -1)
    if valid is not None:
        la, lb = la[valid], lb[valid]
    kl = F.kl_div(lb, la, log_target=True, reduction="sum") + F.kl_div(
        la, lb, log_target=True, reduction="sum"
    )
    return 0.5 * kl


@contextlib.contextmanager
def _weights_in(params: list[torch.Tensor], weights: list[torch.Tensor] | None):
    """Temporarily load `weights` into `params` (None keeps the live weights)."""
    if weights is None:
        yield
        return
    backup = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, w in zip(params, weights):
            p.copy_(w)
    try:
        yield
    finally:
        with torch.no_grad():
            for p, b in zip(params, backup):
                p.copy_(b)


class _Data:
    """Encoded train/val rows and the label bookkeeping for one task."""

    def __init__(self, cfg: TrainConfig, tok, train_rows, val_rows, labels: list[str]):
        self.labels = labels
        self.token = cfg.task == "token-cls"
        self.pad = tok.pad_token_id
        if not self.token:
            self.train_ids, t1 = encode_seq(tok, [r["text"] for r in train_rows], cfg.max_len)
            self.val_ids, t2 = encode_seq(tok, [r["text"] for r in val_rows], cfg.max_len)
            idx = {l: i for i, l in enumerate(labels)}
            self.train_y = [idx[str(r["label"])] for r in train_rows]
            self.val_y = self.val_words = None
        else:
            tag2id = {t: i for i, t in enumerate(labels)}
            self.train_ids, self.train_y, _, t1 = encode_tokens(
                tok, [r["tokens"] for r in train_rows], [r["tags"] for r in train_rows],
                tag2id, cfg.max_len, cfg.label_align,
            )
            # val tags only decide which positions carry a label; unseen tags are mapped to "O"
            val_tags = [[t if t in tag2id else "O" for t in r["tags"]] for r in val_rows]
            self.val_ids, self.val_y, self.val_words, t2 = encode_tokens(
                tok, [r["tokens"] for r in val_rows], val_tags, tag2id, cfg.max_len, cfg.label_align,
            )
            self.val_n_words = [len(r["tokens"]) for r in val_rows]
        self.n_truncated = t1 + t2


def _build_model(cfg: TrainConfig, profile, n_out: int, device: str) -> nn.Module:
    backbone = load_backbone(profile, device, cfg.hidden_dropout, cfg.attn_dropout)
    if cfg.reinit_top:
        reinit_top(backbone, cfg.reinit_top)
    if cfg.freeze_embed:
        frozen = [p for n, p in backbone.named_parameters() if "embed" in n]
        if not frozen:
            raise RuntimeError("freeze_embed matched no parameters")
        for p in frozen:
            p.requires_grad_(False)
    if cfg.lora_r > 0:
        backbone = apply_lora(backbone, profile, cfg.lora_r, cfg.lora_alpha)
    cls = TokenClassifier if cfg.task == "token-cls" else SeqClassifier
    model = cls(backbone, n_out, cfg).to(device)
    if profile.grad_checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


@torch.no_grad()
def _predict(model, cfg: TrainConfig, data: _Data, device: str):
    model.eval()
    n = len(data.val_ids)
    order = sorted(range(n), key=lambda i: len(data.val_ids[i]))
    if not data.token:
        probs = np.zeros((n, len(data.labels)), dtype=np.float64)
    else:
        out = [["O"] * data.val_n_words[i] for i in range(n)]
    for b in range(0, n, _EVAL_BATCH):
        idx = order[b : b + _EVAL_BATCH]
        if not data.token:
            ids, mask = collate([data.val_ids[i] for i in idx], data.pad, device)
            with _autocast(device, cfg):
                logits = model(ids, mask)
            probs[idx] = torch.softmax(logits.float(), -1).cpu().double().numpy()
            continue
        ids, mask, y = collate(
            [data.val_ids[i] for i in idx], data.pad, device, [data.val_y[i] for i in idx]
        )
        with _autocast(device, cfg):
            logits = model(ids, mask)
        lab = y != -100
        if model.crf is not None:
            paths = model.decode(logits.float(), lab)
        else:
            best = logits.argmax(-1)
            paths = [best[r][lab[r]].tolist() for r in range(len(idx))]
        for r, i in enumerate(idx):
            positions = [j for j, v in enumerate(data.val_y[i]) if v != -100]
            seen: set[int] = set()
            for j, tag in zip(positions, paths[r]):
                w = data.val_words[i][j]
                if w not in seen:
                    seen.add(w)
                    out[i][w] = data.labels[tag]
    model.train()
    if not data.token:
        return probs / probs.sum(1, keepdims=True)  # exact row sums for sklearn
    return out


def _class_weights(cfg: TrainConfig, data: _Data, pos: int):
    n = len(data.labels)
    if not data.token:
        return class_weights(cfg, data.train_y, n, pos)
    if cfg.class_weight == "balanced":
        flat = [t for row in data.train_y for t in row if t != -100]
        return class_weights(cfg, flat, n, pos)
    if cfg.class_weight == 1.0:
        return None
    w = torch.ones(n)  # numeric weight applies to every entity (non-"O") tag
    w[[i for i, t in enumerate(data.labels) if t != "O"]] = float(cfg.class_weight)
    return w


def train_and_predict(
    cfg: TrainConfig, profile, train_rows: list[dict], val_rows: list[dict], labels: list[str],
    select: Callable, device: str, keep_model: bool = False, direction: str = "max", pos: int = 1,
) -> TrainResult:
    if cfg.ema and cfg.swa:
        raise ValueError("ema and swa cannot both be enabled")
    if cfg.fgm_eps and (cfg.freeze_embed or cfg.lora_r > 0):
        raise ValueError("fgm_eps needs trainable word embeddings (not with freeze_embed or lora_r)")
    from transformers import AutoTokenizer

    cfg = runtime_config(cfg, profile, device)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(
        profile.hf_id, revision=profile.revision, trust_remote_code=profile.trust_remote_code
    )
    model = _build_model(cfg, profile, len(labels), device)
    data = _Data(cfg, tok, train_rows, val_rows, labels)
    token = data.token
    n_train = len(data.train_ids)
    if not n_train:
        raise ValueError("no training rows")
    batch, micro = cfg.batch, cfg.micro
    oversample = 1 if token else cfg.oversample
    epoch_len = _epoch_len(data.train_y, oversample) if not token else n_train
    steps = cfg.max_steps or round(cfg.epochs * epoch_len / batch * cfg.step_fraction)
    cap = test_env("FINETUNE_KIT_MAX_STEPS")
    if cap:
        steps = min(steps, int(cap))
    steps = max(1, steps)
    eval_steps = max(1, round(cfg.eval_every * epoch_len / batch))
    weights = _class_weights(cfg, data, pos)
    o_id = labels.index("O") if token else 0

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        param_groups(model, cfg), betas=(cfg.beta1, cfg.beta2), eps=cfg.eps
    )
    shadow = [p.detach().clone() for p in trainable] if cfg.ema else None
    swa_start = steps - max(1, round(0.25 * steps)) if cfg.swa else None
    swa_avg, swa_n = None, 0
    emb = model.backbone.get_input_embeddings().weight if cfg.fgm_eps else None

    def task_loss(logits, y):
        if not token:
            return seq_loss(logits, y, cfg, weights, pos)
        if model.crf is not None:
            return model.nll(logits, y)
        return token_loss(logits, y, cfg, weights, o_id)

    def micro_loss(ids, mask, y):
        def fwd():
            with _autocast(device, cfg):
                return model(ids, mask).float()

        a = fwd()
        if not cfg.rdrop_alpha:
            return task_loss(a, y) / batch
        b = fwd()
        valid = (y != -100) if token else None
        kl = _sym_kl(a, b, valid)
        return (0.5 * (task_loss(a, y) + task_loss(b, y)) + cfg.rdrop_alpha * kl) / batch

    def backward_all(chunks) -> float:
        total = 0.0
        for ids, mask, y in chunks:
            loss = micro_loss(ids, mask, y)
            loss.backward()
            total += loss.item()
        return total

    def make_chunks(idx):
        out = []
        for m in range(0, len(idx), micro):
            sub = idx[m : m + micro]
            rows = [data.train_ids[i] for i in sub]
            if token:
                out.append(collate(rows, data.pad, device, [data.train_y[i] for i in sub]))
            else:
                ids, mask = collate(rows, data.pad, device)
                out.append((ids, mask, torch.tensor([data.train_y[i] for i in sub], device=device)))
        return out

    best_score, best_out, best_w, best_epoch = None, None, None, 0.0

    def evaluate(step: int, live: list[torch.Tensor] | None):
        nonlocal best_score, best_out, best_w, best_epoch
        with _weights_in(trainable, live):
            out = _predict(model, cfg, data, device)
            score = float(select(out))
            better = best_score is None or (
                not math.isnan(score)
                and (math.isnan(best_score) or (score > best_score if direction == "max" else score < best_score))
            )
            if better:
                best_score, best_out = score, out
                best_w = [p.detach().cpu().clone() for p in trainable]
                best_epoch = (step + 1) * batch / epoch_len

    rng = random.Random(cfg.seed)
    order: list[int] = []
    model.train()
    t0 = time.time()
    for step in range(steps):
        f = lr_factor(step, steps, cfg)
        for g in opt.param_groups:
            g["lr"] = cfg.lr * g["base_mult"] * f
        while len(order) < batch:
            order += (
                epoch_order(data.train_y, oversample, rng) if not token
                else rng.sample(range(n_train), n_train)
            )
        idx, order = order[:batch], order[batch:]
        chunks = make_chunks(idx)
        loss_sum = backward_all(chunks)
        if not math.isfinite(loss_sum):
            raise RuntimeError(f"non-finite loss at step {step}")
        if emb is not None and emb.grad is not None:
            norm = emb.grad.norm()
            if norm > 0:
                delta = cfg.fgm_eps * emb.grad / norm
                with torch.no_grad():
                    emb.add_(delta)
                backward_all(chunks)
                with torch.no_grad():
                    emb.sub_(delta)
        nn.utils.clip_grad_norm_(trainable, cfg.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            if shadow is not None:
                torch._foreach_lerp_(shadow, [p.detach() for p in trainable], 1.0 - cfg.ema)
            if swa_start is not None and step >= swa_start:
                swa_n += 1
                if swa_avg is None:
                    swa_avg = [p.detach().clone() for p in trainable]
                else:  # equal-weight running mean
                    torch._foreach_lerp_(swa_avg, [p.detach() for p in trainable], 1.0 / swa_n)
        last = step == steps - 1
        if cfg.swa:
            if last:
                evaluate(step, swa_avg)
        elif last or (step + 1) % eval_steps == 0:
            evaluate(step, shadow)
    train_s = time.time() - t0
    peak = _peak_gb(device)

    with torch.no_grad():
        for p, w in zip(trainable, best_w):
            p.copy_(w.to(p.device))
    result = TrainResult(
        val_out=best_out, best_epoch=best_epoch, train_s=train_s, peak_gb=peak,
        n_truncated=data.n_truncated, steps=steps, model=model if keep_model else None,
    )
    if not keep_model:
        del model, opt
        if device == "cuda":
            torch.cuda.empty_cache()
    return result
