"""Model profiles: shipped YAML, project overrides, and generic profiles for new models."""
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

SHIPPED = Path(__file__).parent / "profiles" / "models.yaml"
BERT_DROPOUT = {"hidden": ["hidden_dropout_prob"], "attn": ["attention_probs_dropout_prob"]}
BERT_LORA = ["query", "key", "value", "dense"]


@dataclass
class ModelProfile:
    key: str
    hf_id: str
    revision: str
    tasks: list[str]
    lr_default: float
    lr_range: tuple[float, float]
    max_len: int
    batch: int
    micro: int
    force_fp32_on_mps: bool = False
    grad_checkpointing: bool = False
    trust_remote_code: bool = False
    dropout_keys: dict[str, list[str]] = field(default_factory=dict)
    lora_targets: list[str] = field(default_factory=list)
    notes: str = ""


def _from_dict(key: str, d: dict) -> ModelProfile:
    names = {f.name for f in fields(ModelProfile)} - {"key"}
    extra = set(d) - names
    if extra:
        raise ValueError(f"profile {key!r}: unknown field(s) {sorted(extra)}")
    d = dict(d)
    d["lr_range"] = tuple(float(x) for x in d["lr_range"])
    d["lr_default"] = float(d["lr_default"])
    return ModelProfile(key=key, **d)


def _to_dict(p: ModelProfile) -> dict:
    d = {f.name: getattr(p, f.name) for f in fields(p) if f.name != "key"}
    d["lr_range"] = list(p.lr_range)
    return d


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of profile key to fields")
    return data


def _test_profiles() -> dict[str, ModelProfile]:
    common = dict(
        tasks=["seq-cls", "token-cls"], lr_default=1e-3, lr_range=(1e-4, 1e-2), max_len=64,
        batch=8, micro=4, dropout_keys=BERT_DROPOUT, lora_targets=BERT_LORA, revision="main",
        notes="test profile",
    )
    return {
        "tiny-bert": ModelProfile(key="tiny-bert", hf_id="hf-internal-testing/tiny-random-bert", **common),
        "tiny-roberta": ModelProfile(
            key="tiny-roberta", hf_id="hf-internal-testing/tiny-random-RobertaModel", **common
        ),
    }


def load_profiles(project_root: Path | None) -> dict[str, ModelProfile]:
    raw = _read_yaml(SHIPPED)
    if project_root is not None:
        raw.update(_read_yaml(Path(project_root) / ".finetune-kit" / "models.yaml"))
    profiles = {k: _from_dict(k, v) for k, v in raw.items()}
    if os.environ.get("FINETUNE_KIT_TEST_PROFILES") == "1":
        profiles.update(_test_profiles())
    return profiles


def _slug(hf_id: str) -> str:
    return hf_id.rsplit("/", 1)[-1].lower().replace("_", "-").replace(".", "-")


def add_profile(
    project_root: Path, hf_id: str, trust_remote_code: bool, config: dict | None = None
) -> ModelProfile:
    """Register a generic profile in `<project_root>/.finetune-kit/models.yaml`.

    `config` is the model's HF config.json as a dict; when omitted, config.json (and the
    current commit sha) is fetched from the Hub. No weights are downloaded.
    """
    revision = "main"
    if config is None:
        import json

        from huggingface_hub import HfApi, hf_hub_download

        revision = HfApi().model_info(hf_id).sha or "main"
        config = json.loads(Path(hf_hub_download(hf_id, "config.json", revision=revision)).read_text())
    mpe = int(config.get("max_position_embeddings") or 512)
    max_len = mpe if config.get("model_type") == "modernbert" and mpe > 512 else min(mpe, 512)
    bert_like = "hidden_dropout_prob" in config and "attention_probs_dropout_prob" in config
    profile = ModelProfile(
        key=_slug(hf_id), hf_id=hf_id, revision=revision, tasks=["seq-cls", "token-cls"],
        lr_default=3e-5, lr_range=(1e-5, 1e-4), max_len=max_len, batch=8, micro=4,
        trust_remote_code=trust_remote_code,
        dropout_keys={k: list(v) for k, v in BERT_DROPOUT.items()} if bert_like else {},
        lora_targets=list(BERT_LORA) if bert_like else [],
        notes="generic: lr-scan required",
    )
    path = Path(project_root) / ".finetune-kit" / "models.yaml"
    data = _read_yaml(path)
    data[profile.key] = _to_dict(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return profile
