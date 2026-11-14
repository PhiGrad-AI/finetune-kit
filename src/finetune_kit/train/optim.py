"""Optimizer parameter groups (LLRD, head lr multiplier) and the lr schedule."""
import math
import re

_LAYER = re.compile(r"\.layers?\.(\d+)\.")
_HEAD_PREFIXES = ("head.", "crf.", "pool.")


def param_groups(model, cfg) -> list[dict]:
    n_layers = model.backbone.config.num_hidden_layers
    groups: dict[tuple, list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(_HEAD_PREFIXES):
            depth_mult = cfg.head_lr_mult
        elif cfg.llrd == 1.0:
            depth_mult = 1.0
        else:
            m = _LAYER.search(name)
            layer = int(m.group(1)) + 1 if m else (0 if "embed" in name else n_layers)
            depth_mult = cfg.llrd ** (n_layers - layer)
        wd = cfg.weight_decay if p.ndim >= 2 else 0.0
        groups.setdefault((round(depth_mult, 6), wd), []).append(p)
    return [
        {"params": ps, "lr": cfg.lr * mult, "weight_decay": wd, "base_mult": mult}
        for (mult, wd), ps in groups.items()
    ]


def lr_factor(step: int, steps: int, cfg) -> float:
    warm = max(1, int(cfg.warmup * steps))
    if step < warm:
        return (step + 1) / warm
    prog = (step - warm) / max(1, steps - warm)
    if cfg.schedule == "cosine":
        return 0.5 * (1 + math.cos(math.pi * prog))
    if cfg.schedule == "constant":
        return 1.0
    return max(0.0, 1 - prog)
