"""TrainConfig: every catalog key plus the fixed trainer settings."""
from typing import Any

from pydantic import ConfigDict, create_model

from finetune_kit.catalog import ENTRIES, names
from finetune_kit.profiles import ModelProfile

_SPECIAL_TYPES = {
    "class_weight": float | str,
    "hidden_dropout": float | None,
    "attn_dropout": float | None,
}
_KIND_TYPES = {"float": float, "int": int, "choice": Any, "bool": bool}


def _catalog_fields() -> dict:
    return {
        e.name: (_SPECIAL_TYPES.get(e.name) or _KIND_TYPES[e.kind], e.default)
        for e in ENTRIES.values()
    }


_CatalogFields = create_model(
    "_CatalogFields", __config__=ConfigDict(extra="forbid"), **_catalog_fields()
)


class TrainConfig(_CatalogFields):
    task: str
    model: str
    seed: int
    micro: int
    beta1: float = 0.9
    eps: float = 1e-6
    eval_every: float = 0.5
    max_steps: int | None = None
    step_fraction: float = 1.0  # LR scan: run only this fraction of the schedule


def defaults_for(profile: ModelProfile, task: str) -> dict:
    """Default value of every catalog key valid for `task`; lr/max_len/batch come from the profile."""
    d = {n: ENTRIES[n].default for n in names(task)}
    d.update(lr=profile.lr_default, max_len=profile.max_len, batch=profile.batch)
    return d


def build_config(overrides: dict, profile: ModelProfile, task: str, seed: int) -> TrainConfig:
    return TrainConfig(
        **{**defaults_for(profile, task), **overrides},
        task=task, model=profile.key, seed=seed, micro=profile.micro,
    )


def nondefault_count(overrides: dict, defaults: dict) -> int:
    return sum(1 for k, v in overrides.items() if k not in defaults or v != defaults[k])
