"""Hyperparameter catalog: hard limits, recommendations and config validation."""
import math
from dataclasses import dataclass
from typing import Any, Literal

from finetune_kit.profiles import ModelProfile

SEQ, TOK = "seq-cls", "token-cls"
BOTH = (SEQ, TOK)
LOSSES = {SEQ: ["ce", "focal", "asl", "rank", "ce+rank"], TOK: ["ce", "focal", "dice"]}
RECALL_FIRST = {"skip_at_recall", "precision_at_recall", "pauc_hi"}
RANKING = {"pauc_hi", "skip_at_recall", "precision_at_recall", "recall_at_precision", "roc_auc", "pr_auc"}


@dataclass(frozen=True)
class Entry:
    name: str
    group: str
    tasks: tuple[str, ...]
    kind: Literal["float", "int", "choice", "bool"]
    hard: tuple | list
    log: bool
    help: str
    default: Any = None


def _e(name, group, kind, hard, help, default, tasks=BOTH, log=False):
    return Entry(name, group, tuple(tasks), kind, hard, log, help, default)


_LIST = [
    _e("lr", "optimizer", "float", (1e-7, 1e-3), "peak learning rate", 3e-5, log=True),
    _e("llrd", "optimizer", "float", (0.5, 1.0), "layer-wise lr decay factor (1 = off)", 1.0),
    _e("head_lr_mult", "optimizer", "float", (1, 100), "lr multiplier for the classifier head", 1.0),
    _e("warmup", "schedule", "float", (0, 0.5), "warmup fraction of total steps", 0.05),
    _e("schedule", "schedule", "choice", ["linear", "cosine", "constant"], "lr schedule", "linear"),
    _e("weight_decay", "optimizer", "float", (0, 0.5), "AdamW weight decay", 0.01),
    _e("epochs", "schedule", "float", (0.5, 20), "training epochs", 2),
    _e("batch", "schedule", "choice", [4, 8, 16, 32, 64], "effective batch size", 8),
    _e("clip", "optimizer", "float", (0.1, 10), "gradient clipping norm", 1.0),
    _e("beta2", "optimizer", "float", (0.95, 0.9999), "Adam beta2", 0.999),
    _e("pooling", "head", "choice", ["cls", "mean", "attention", "weighted_layers"],
       "sequence pooling", "cls", tasks=(SEQ,)),
    _e("pool_layers", "head", "int", (2, 12), "layers used by weighted_layers pooling", 4, tasks=(SEQ,)),
    _e("head", "head", "choice", ["linear", "mlp"], "classifier head type", "linear"),
    _e("dropout", "regularization", "float", (0, 0.6), "classifier head dropout", 0.1),
    _e("msd_samples", "regularization", "int", (1, 8), "multi-sample dropout samples (1 = off)", 1),
    _e("hidden_dropout", "regularization", "float", (0, 0.3), "backbone hidden dropout (default: model's own)", None),
    _e("attn_dropout", "regularization", "float", (0, 0.3), "backbone attention dropout (default: model's own)", None),
    _e("rdrop_alpha", "regularization", "float", (0, 10), "R-Drop KL weight (0 = off)", 0.0),
    _e("ema", "regularization", "float", (0.9, 0.9999), "EMA decay (0 = off)", 0.0),
    _e("swa", "regularization", "bool", [True, False], "stochastic weight averaging", False),
    _e("reinit_top", "regularization", "int", (0, 6), "re-initialise the top N layers", 0),
    _e("freeze_embed", "regularization", "bool", [True, False], "freeze embeddings", False),
    _e("fgm_eps", "regularization", "float", (0, 5), "FGM adversarial epsilon (0 = off)", 0.0),
    _e("lora_r", "adapter", "choice", [0, 4, 8, 16, 32, 64], "LoRA rank (0 = off)", 0),
    _e("lora_alpha", "adapter", "float", (1, 128), "LoRA alpha (needs lora_r > 0)", 16),
    _e("loss", "loss", "choice", [], "training loss", "ce"),
    _e("class_weight", "loss", "float", (1, 20), 'positive-class weight or "balanced"', 1.0),
    _e("focal_gamma", "loss", "float", (0, 5), "focal loss gamma", 2.0),
    _e("label_smoothing", "loss", "float", (0, 0.3), "label smoothing", 0.0),
    _e("rank_weight", "loss", "float", (0, 10), "rank loss weight", 1.0, tasks=(SEQ,)),
    _e("asl_gamma_neg", "loss", "float", (0, 8), "ASL negative gamma", 4.0, tasks=(SEQ,)),
    _e("asl_gamma_pos", "loss", "float", (0, 5), "ASL positive gamma", 0.0, tasks=(SEQ,)),
    _e("asl_clip", "loss", "float", (0, 0.2), "ASL probability margin", 0.05, tasks=(SEQ,)),
    _e("dice_smooth", "loss", "float", (0.01, 10), "dice smoothing", 1.0, tasks=(TOK,), log=True),
    _e("dice_ce_weight", "loss", "float", (0, 1), "CE share in dice+CE", 0.5, tasks=(TOK,)),
    _e("oversample", "data", "int", (1, 10), "minority-class sampling factor (1 = off)", 1, tasks=(SEQ,)),
    _e("max_len", "data", "int", (32, 512), "max tokens per example", 512),
    _e("precision", "system", "choice", ["fp32", "bf16"], "training precision", "fp32"),
    _e("crf", "head", "bool", [True, False], "CRF output layer", False, tasks=(TOK,)),
    _e("label_align", "data", "choice", ["first", "all"], "which sub-tokens carry the label", "first", tasks=(TOK,)),
]
ENTRIES: dict[str, Entry] = {e.name: e for e in _LIST}


def names(task: str) -> list[str]:
    return [n for n, e in ENTRIES.items() if task in e.tasks]


def fmt(v: float) -> str:
    """Format a limit the way the docs write it: 1e-3, 0.5, 20."""
    if v != 0 and abs(v) < 1e-2:
        mant, exp = f"{v:.0e}".split("e")
        return f"{mant}e{int(exp)}"
    return f"{v:g}"


def recommend(report: dict, objective: str, task: str, max_len: int = 512) -> dict[str, str]:
    rec = {"lr": "most sensitive", "llrd": "largest single gain in past tuning"}
    share = report.get("minority_share")
    if share is not None and share < 0.3:
        rec["class_weight"] = f"minority class is {share:.0%} of rows"
        rec["loss:focal"] = f"minority class is {share:.0%} of rows"
    if task == SEQ and objective in RECALL_FIRST:
        rec["loss:asl"] = "pushes harder on missed positives"
    ent = report.get("entity_token_share")
    if task == TOK and ent is not None and ent < 0.1:
        rec["loss:dice"] = "entities are rare"
    if task == SEQ and objective in RANKING:
        rec["loss:ce+rank"] = f"{objective} rewards ranking quality"
    p95 = (report.get("token_len") or {}).get("p95")
    if p95 is not None and p95 < 0.5 * max_len:
        rec["max_len"] = "shorter runs"
    return rec


def listing(task: str, profile: ModelProfile, report: dict, objective: str) -> list[dict]:
    rec = recommend(report, objective, task, profile.max_len)
    overrides = {"lr": profile.lr_default, "max_len": profile.max_len, "batch": profile.batch}
    loss_reasons = [f"{k.split(':', 1)[1]}: {v}" for k, v in rec.items() if k.startswith("loss:")]
    rows = []
    for n in names(task):
        e = ENTRIES[n]
        reason = rec.get(n)
        if n == "loss" and loss_reasons:
            reason = "; ".join(loss_reasons)
        rows.append({
            "name": n, "group": e.group, "default": overrides.get(n, e.default), "help": e.help,
            "recommended": reason is not None, "reason": reason,
        })
    return rows


LOSS_HELP = {
    "ce": "standard cross-entropy (the default)",
    "focal": "focuses training on hard examples; helps with imbalanced labels",
    "asl": "asymmetric loss; pushes harder on missed positives",
    "rank": "rewards ranking positives above negatives",
    "ce+rank": "cross-entropy plus ranking; suits AUC-style metrics",
    "dice": "overlap-based; helps when entities are rare (not with crf)",
}


def loss_options(task: str, profile: ModelProfile, report: dict, objective: str) -> list[dict]:
    """The losses research may be allowed to try for this task, with recommendations."""
    rec = recommend(report, objective, task, profile.max_len)
    rows = []
    for name in LOSSES[task]:
        reason = rec.get(f"loss:{name}")
        rows.append({
            "name": name, "help": LOSS_HELP[name], "always": name == "ce",
            "recommended": reason is not None, "reason": reason,
        })
    return rows


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_config(
    config: dict, tunable: list[str], losses: list[str], defaults: dict, task: str,
    profile: ModelProfile | None = None,
) -> list[str]:
    errs: list[str] = []
    allowed_losses = [l for l in losses if l in LOSSES[task]]
    for key, val in config.items():
        e = ENTRIES.get(key)
        if e is None or task not in e.tasks:
            errs.append(f"unknown hyperparameter {key!r} for {task}")
            continue
        has_default = key in defaults
        if has_default and val == defaults[key]:
            continue  # keys equal to default are always OK
        if e.kind in ("float", "int"):
            if key == "class_weight" and val == "balanced":
                pass
            elif not _is_number(val):
                errs.append(f"{key}: expected a number, got {val!r}")
                continue
            elif not math.isfinite(val):
                errs.append(f"{key}: expected a finite number, got {val!r}")
                continue
            elif e.kind == "int" and val != int(val):
                errs.append(f"{key}: expected a whole number, got {val!r}")
                continue
        elif e.kind == "bool" and not isinstance(val, bool):
            errs.append(f"{key}: expected true or false, got {val!r}")
            continue
        if key not in tunable:
            errs.append(f"{key} is not tunable (locked at {defaults.get(key)!r}); open it in another round (steps choose-loss, then choose-hyperparameters)")
            continue
        errs.extend(_check_value(e, val, allowed_losses, profile))
        errs.extend(_check_model_support(key, val, profile))
    if "lora_alpha" in config and config["lora_alpha"] != defaults.get("lora_alpha"):
        if config.get("lora_r", defaults.get("lora_r", 0)) == 0:
            errs.append("lora_alpha requires lora_r > 0")
    if task == TOK and config.get("loss", defaults.get("loss")) == "dice" and config.get(
        "crf", defaults.get("crf", False)
    ):
        errs.append("loss dice is not allowed with crf: true")
    return errs


def _check_value(e: Entry, val, allowed_losses: list[str], profile: ModelProfile | None) -> list[str]:
    key = e.name
    if key == "loss":
        return [] if val in allowed_losses else [f"loss {val!r} not allowed; choose one of {allowed_losses}"]
    if e.kind == "choice":
        return [] if val in e.hard else [f"{key}={val!r} not allowed; choose one of {list(e.hard)}"]
    if e.kind == "bool":
        return []
    if key == "class_weight" and val == "balanced":
        return []
    lo, hi = e.hard
    if key == "max_len" and profile is not None:
        hi = profile.max_len
    if key == "ema" and val == 0:
        return []
    if not lo <= val <= hi:
        return [f"{key}={val} outside hard limits [{fmt(lo)}, {fmt(hi)}]"]
    return []


def _check_model_support(key: str, val, profile: ModelProfile | None) -> list[str]:
    if profile is None:
        return []
    side = {"hidden_dropout": "hidden", "attn_dropout": "attn"}.get(key)
    if side and not profile.dropout_keys.get(side):
        return [f"{key} is not supported for this model (no {side} dropout keys in profile {profile.key!r})"]
    if key == "lora_r" and val != 0 and not profile.lora_targets:
        return [f"lora_r is not supported for this model (no LoRA targets in profile {profile.key!r})"]
    return []
