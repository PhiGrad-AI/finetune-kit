"""Sequence and token classification losses. All return the SUM over the batch."""
import torch
from torch.nn import functional as F


def class_weights(cfg, train_label_ids, n_classes: int, pos: int) -> torch.Tensor | None:
    cw = cfg.class_weight
    if cw == "balanced":
        counts = torch.bincount(torch.as_tensor(list(train_label_ids)), minlength=n_classes).float()
        return counts.sum() / (n_classes * counts.clamp(min=1.0))
    if cw == 1.0:
        return None
    w = torch.ones(n_classes)
    w[pos] = float(cw)
    return w


def _asl(logits, y, cfg):
    onehot = F.one_hot(y, logits.shape[-1]).to(logits.dtype)
    p = torch.sigmoid(logits)
    pm = (p - cfg.asl_clip).clamp(min=0)
    eps = 1e-8
    pos = -(torch.sigmoid(-logits).clamp(min=eps) ** cfg.asl_gamma_pos) * F.logsigmoid(logits)
    neg = -(pm ** cfg.asl_gamma_neg) * torch.log((1 - pm).clamp(min=eps))
    return (onehot * pos + (1 - onehot) * neg).sum()


def seq_loss(logits, y, cfg, weights, pos: int) -> torch.Tensor:
    kind = cfg.loss
    if kind == "asl":
        return _asl(logits, y, cfg)
    w = weights.to(logits.device, logits.dtype) if weights is not None else None
    ce = F.cross_entropy(logits, y, weight=w, reduction="none", label_smoothing=cfg.label_smoothing)
    if kind == "focal":
        pt = torch.softmax(logits, -1).gather(1, y[:, None]).squeeze(1).detach()
        return ((1 - pt) ** cfg.focal_gamma * ce).sum()
    if kind in ("rank", "ce+rank"):
        if logits.shape[-1] != 2:
            raise ValueError(f"loss {kind!r} needs exactly 2 classes, got {logits.shape[-1]}")
        margin = logits[:, pos] - logits[:, 1 - pos]
        up, down = margin[y == pos], margin[y != pos]
        rank = (
            F.softplus(down[None, :] - up[:, None]).mean()
            if len(up) and len(down) else margin.sum() * 0
        )
        if kind == "rank":
            return rank * len(y)  # scaled so batch averaging matches ce
        return ce.sum() + cfg.rank_weight * rank * len(y)
    return ce.sum()


def _soft_dice(logits, labels, o_id: int, smooth: float) -> torch.Tensor:
    p = torch.softmax(logits, -1)
    y = F.one_hot(labels, logits.shape[-1]).to(p.dtype)
    classes = [c for c in range(logits.shape[-1]) if c != o_id]
    inter = (p * y).sum(0)[classes]
    denom = p.sum(0)[classes] + y.sum(0)[classes]
    return ((2 * inter + smooth) / (denom + smooth)).mean()


def token_loss(logits, labels, cfg, weights, o_id: int) -> torch.Tensor:
    flat = logits.reshape(-1, logits.shape[-1])
    lab = labels.reshape(-1)
    valid = lab != -100
    n = int(valid.sum())
    if n == 0:
        return flat.sum() * 0
    w = weights.to(flat.device, flat.dtype) if weights is not None else None
    ce = F.cross_entropy(flat, lab, weight=w, ignore_index=-100, reduction="none",
                         label_smoothing=cfg.label_smoothing)
    if cfg.loss == "focal":
        pt = torch.softmax(flat[valid], -1).gather(1, lab[valid, None]).squeeze(1).detach()
        return ((1 - pt) ** cfg.focal_gamma * ce[valid]).sum()
    ce_sum = ce.sum()
    if cfg.loss == "dice":
        a = cfg.dice_ce_weight
        if a >= 1.0:
            return ce_sum
        dice = _soft_dice(flat[valid], lab[valid], o_id, cfg.dice_smooth)
        return a * ce_sum + (1 - a) * (1 - dice) * n  # scaled by n so it matches the CE sum
    return ce_sum
