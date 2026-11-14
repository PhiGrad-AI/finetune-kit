"""Backbone loading, LoRA, classifier heads and tokenisation helpers."""
import torch
from torch import nn

from finetune_kit.profiles import ModelProfile


def load_backbone(
    profile: ModelProfile, device: str, hidden_dropout: float | None, attn_dropout: float | None
) -> nn.Module:
    from transformers import AutoConfig, AutoModel

    kw = dict(revision=profile.revision, trust_remote_code=profile.trust_remote_code)
    config = AutoConfig.from_pretrained(profile.hf_id, **kw)
    for side, value in (("hidden", hidden_dropout), ("attn", attn_dropout)):
        if value is not None:
            for key in profile.dropout_keys.get(side, []):
                setattr(config, key, value)
    # Weights are always fp32 (bf16 is autocast); force_fp32_on_mps just pins that on mps.
    model = AutoModel.from_pretrained(profile.hf_id, config=config, dtype=torch.float32, **kw)
    return model.to(device)


def apply_lora(backbone: nn.Module, profile: ModelProfile, r: int, alpha: float) -> nn.Module:
    from peft import LoraConfig, get_peft_model

    lcfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=profile.lora_targets, lora_dropout=0.05)
    return get_peft_model(backbone, lcfg)


class _Pool(nn.Module):
    """Pooling parameters; lives under `pool.` so it trains at the head lr."""

    def __init__(self, cfg, hidden: int):
        super().__init__()
        self.kind = cfg.pooling
        if cfg.pooling == "attention":
            self.attn = nn.Linear(hidden, 1)
        if cfg.pooling == "weighted_layers":
            self.layer_weights = nn.Parameter(torch.zeros(cfg.pool_layers))
        self.n_layers = cfg.pool_layers

    def forward(self, out, attention_mask):
        h = out.last_hidden_state
        if self.kind == "mean":
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            return (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        if self.kind == "attention":
            scores = self.attn(h).squeeze(-1).masked_fill(attention_mask == 0, float("-inf"))
            return (torch.softmax(scores, -1).unsqueeze(-1) * h).sum(1)
        if self.kind == "weighted_layers":
            cls = torch.stack([s[:, 0] for s in out.hidden_states[-self.n_layers:]], 1)
            w = torch.softmax(self.layer_weights, 0).to(cls.dtype)
            return (w[None, :, None] * cls).sum(1)
        return h[:, 0]


def _head(cfg, hidden: int, n_out: int) -> nn.Sequential:
    if cfg.head == "mlp":
        return nn.Sequential(
            nn.Dropout(cfg.dropout), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(hidden, n_out),
        )
    return nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(hidden, n_out))


class SeqClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, n_classes: int, cfg):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        if cfg.pooling == "weighted_layers" and cfg.pool_layers > backbone.config.num_hidden_layers:
            raise ValueError(
                f"pool_layers={cfg.pool_layers} exceeds the model's "
                f"{backbone.config.num_hidden_layers} layers"
            )
        self.pool = _Pool(cfg, hidden)
        self.head = _head(cfg, hidden, n_classes)
        self.msd = cfg.msd_samples
        self.needs_hidden = cfg.pooling == "weighted_layers"

    def forward(self, input_ids, attention_mask):
        kw = {"output_hidden_states": True} if self.needs_hidden else {}
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kw)
        pooled = self.pool(out, attention_mask)
        if self.training and self.msd > 1:
            return torch.stack([self.head(pooled) for _ in range(self.msd)]).mean(0)
        return self.head(pooled)


class TokenClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, n_tags: int, cfg):
        super().__init__()
        self.backbone = backbone
        self.head = _head(cfg, backbone.config.hidden_size, n_tags)
        self.crf = None
        if cfg.crf:
            from torchcrf import CRF

            self.crf = CRF(n_tags, batch_first=True)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(h)

    @staticmethod
    def _compact(logits, keep):
        """Left-align the kept positions of each row (CRF needs a mask that starts on)."""
        lengths = keep.sum(1)
        t = int(lengths.max()) if len(lengths) else 0
        order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)[:, :t]
        packed = logits.gather(1, order[:, :, None].expand(-1, -1, logits.shape[-1]))
        mask = torch.arange(t, device=logits.device)[None, :] < lengths[:, None]
        return packed.float(), order, mask, lengths

    def nll(self, logits, labels, mask=None) -> torch.Tensor:
        """CRF negative log-likelihood summed over rows; label -100 positions are skipped."""
        keep = labels != -100
        if mask is not None:
            keep = keep & mask.bool()
        rows = keep.sum(1) > 0
        if not rows.any():
            return logits.sum() * 0
        packed, order, cmask, _ = self._compact(logits[rows], keep[rows])
        tags = labels[rows].gather(1, order).clamp(min=0)
        return -self.crf(packed, tags, mask=cmask, reduction="sum")

    def decode(self, logits, mask) -> list[list[int]]:
        """Viterbi tags for the positions where `mask` is true, one list per row."""
        keep = mask.bool()
        out: list[list[int]] = [[] for _ in range(len(keep))]
        rows = keep.sum(1) > 0
        if not rows.any():
            return out
        packed, _, cmask, _ = self._compact(logits[rows], keep[rows])
        paths = self.crf.decode(packed, mask=cmask)
        for i, path in zip(rows.nonzero().flatten().tolist(), paths):
            out[i] = path
        return out


def _num_special(tok) -> int:
    return tok.num_special_tokens_to_add(pair=False)


def encode_seq(tok, texts: list[str], max_len: int) -> tuple[list[list[int]], int]:
    ids = tok(texts, truncation=True, max_length=max_len, verbose=False)["input_ids"]
    full = tok(texts, add_special_tokens=False, verbose=False)["input_ids"]
    n_trunc = sum(1 for f in full if len(f) + _num_special(tok) > max_len)
    return ids, n_trunc


def encode_tokens(
    tok, tokens: list[list[str]], tags: list[list[str]], tag2id: dict[str, int], max_len: int,
    align: str,
) -> tuple[list[list[int]], list[list[int]], list[list[int | None]], int]:
    enc = tok(tokens, is_split_into_words=True, truncation=True, max_length=max_len, verbose=False)
    full = tok(tokens, is_split_into_words=True, add_special_tokens=False, verbose=False)
    n_trunc = sum(1 for f in full["input_ids"] if len(f) + _num_special(tok) > max_len)
    all_ids, all_labels, all_words = [], [], []
    for i, ids in enumerate(enc["input_ids"]):
        words = enc.word_ids(i)
        labels, prev = [], None
        for w in words:
            if w is None:
                labels.append(-100)
            elif w != prev:
                labels.append(tag2id[tags[i][w]])
            elif align == "all":
                t = tags[i][w]
                if t.startswith("B-") and "I-" + t[2:] in tag2id:
                    t = "I-" + t[2:]
                labels.append(tag2id[t])
            else:
                labels.append(-100)
            prev = w
        all_ids.append(ids)
        all_labels.append(labels)
        all_words.append(words)
    return all_ids, all_labels, all_words, n_trunc


def collate(batch_ids: list[list[int]], pad_id: int, device, labels: list[list[int]] | None = None):
    """Pad to the longest row. Returns (ids, mask) or (ids, mask, labels) when labels are given."""
    n = max(len(r) for r in batch_ids)
    ids = torch.full((len(batch_ids), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch_ids), n), dtype=torch.long)
    for i, r in enumerate(batch_ids):
        ids[i, : len(r)] = torch.tensor(r)
        mask[i, : len(r)] = 1
    if labels is None:
        return ids.to(device), mask.to(device)
    y = torch.full((len(batch_ids), n), -100, dtype=torch.long)
    for i, r in enumerate(labels):
        y[i, : len(r)] = torch.tensor(r)
    return ids.to(device), mask.to(device), y.to(device)
