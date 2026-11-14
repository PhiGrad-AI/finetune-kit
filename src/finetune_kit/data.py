"""Data loading, validation, checks, grouped/stratified splits and the test vault."""
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from finetune_kit.paths import vault_dir

_TAG_RE = re.compile(r"^(O|[BIES]-.+)$")
_RECALL_METRICS = {"pauc_hi", "skip_at_recall", "precision_at_recall"}


class DataError(ValueError):
    pass


def validate_rows(
    rows: list[dict], task: str, where: str, lines: list[int] | None = None
) -> list[dict]:
    """Per-row checks. `where` is the file name; messages say `line N` when `lines` is given, else `row N`."""
    seen = set()
    for i, row in enumerate(rows):
        here = f"{where} line {lines[i]}" if lines else f"{where} row {i + 1}"
        required = ("id", "text", "label") if task == "seq-cls" else ("id", "tokens", "tags")
        for key in required:
            if key not in row:
                raise DataError(f"{here}: missing field {key!r}")
        if task == "seq-cls":
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise DataError(f"{here}: text must be a non-empty string")
            if row["label"] is None or row["label"] == "":
                raise DataError(f"{here}: label must not be {'null' if lines else 'empty'}")
        else:
            if not isinstance(row["tokens"], list) or not isinstance(row["tags"], list):
                raise DataError(f"{here}: tokens and tags must be lists")
            if not all(isinstance(t, str) for t in row["tokens"]):
                raise DataError(f"{here}: tokens must be a list of strings")
            if len(row["tokens"]) != len(row["tags"]):
                raise DataError(
                    f"{here}: tokens/tags length mismatch "
                    f"({len(row['tokens'])} vs {len(row['tags'])})"
                )
            for tag in row["tags"]:
                if not isinstance(tag, str) or not _TAG_RE.match(tag):
                    raise DataError(f"{here}: invalid BIO/BIOES tag {tag!r}")
        if row["id"] in seen:
            raise DataError(f"{here}: duplicate id {row['id']!r}")
        seen.add(row["id"])
    return rows


def load_rows(path: Path, task: str, columns: dict | None = None, role: str = "train") -> list[dict]:
    from finetune_kit import sources

    path = Path(path)
    suffix = path.suffix.lower()
    columns = columns or {}
    if suffix in (".csv", ".tsv") and task != "seq-cls":
        raise DataError(f"{path.name}: CSV/TSV support seq-cls only; use JSONL or Parquet for {task}")
    if suffix == ".jsonl":
        numbered = sources.read_jsonl(path)
        lines, rows = [n for n, _ in numbered], [r for _, r in numbered]
        if columns:  # mapping keeps row order and count, so line numbers still line up
            rows = sources.normalize(rows, task, columns, role, path.name)
        return validate_rows(rows, task, path.name, lines)
    raw = sources.read_table(path)
    return validate_rows(sources.normalize(raw, task, columns, role, path.name), task, path.name)


def label_map(train_rows: list[dict], task: str) -> list[str]:
    if task == "seq-cls":
        return sorted({str(r["label"]) for r in train_rows})
    return sorted({t for r in train_rows for t in r["tags"]} | {"O"})


def _text(row: dict) -> str:
    return row["text"] if "text" in row else " ".join(row["tokens"])


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _percentile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    return sorted_vals[min(len(sorted_vals) - 1, math.ceil(q * len(sorted_vals)) - 1)]


def _entity_type(tag: str) -> str:
    return tag.split("-", 1)[1]


def _label_counts(rows: list[dict], task: str) -> dict[str, int]:
    if task == "seq-cls":
        return dict(Counter(str(r["label"]) for r in rows))
    return dict(Counter(_entity_type(t) for r in rows for t in r["tags"] if t != "O" and t[0] in "BS"))


def check(exp_dir: Path, task: str, max_len: int, tokenizer, rows_by_split: dict[str, list[dict]]) -> dict:
    """Data-only report. Val-size warnings are added by the CLI via val_size_warning."""
    warnings: list[str] = []
    errors: list[str] = []
    train = rows_by_split.get("train", [])
    if not train:
        errors.append("train split is empty")
    if rows_by_split.get("val") is not None and not rows_by_split["val"]:
        errors.append("val split is empty")

    label_dist = {s: _label_counts(rows, task) for s, rows in rows_by_split.items()}
    train_counts = label_dist.get("train", {})
    total = sum(train_counts.values())
    minority_share = round(min(train_counts.values()) / total, 4) if total else 0.0
    if total and minority_share < 0.05:
        warnings.append(f"minority class is {minority_share:.1%} of train; consider class weights")

    all_rows = [(s, r) for s, rows in rows_by_split.items() for r in rows]
    lengths: list[int] = []
    if all_rows:
        ids = tokenizer([_text(r) for _, r in all_rows], add_special_tokens=True)["input_ids"]
        lengths = sorted(len(x) for x in ids)
    n_over = sum(1 for n in lengths if n > max_len)
    if n_over:
        warnings.append(f"{n_over} rows exceed max_len {max_len} and will be truncated")

    texts = Counter(_text(r) for _, r in all_rows)
    n_exact = sum(c - 1 for c in texts.values() if c > 1)
    if n_exact:
        warnings.append(f"{n_exact} exact duplicate rows")
    norm_splits: dict[str, set[str]] = defaultdict(set)
    for s, r in all_rows:
        norm_splits[_norm(_text(r))].add(s)
    n_cross = sum(1 for v in norm_splits.values() if len(v) > 1)
    if n_cross:
        warnings.append(f"{n_cross} texts appear in more than one split (leakage)")

    group_splits: dict[str, set[str]] = defaultdict(set)
    for s, r in all_rows:
        if r.get("group_id") is not None:
            group_splits[str(r["group_id"])].add(s)
    n_leak = sum(1 for v in group_splits.values() if len(v) > 1)
    if n_leak:
        warnings.append(f"{n_leak} group_ids appear in more than one split (leakage)")

    train_labels = set(train_counts)
    unseen = sorted(
        {l for s, c in label_dist.items() if s != "train" for l in c if l not in train_labels}
    )
    if unseen:
        warnings.append(f"labels in val/test but not train: {unseen}")

    report = {
        "n_rows": {s: len(rows) for s, rows in rows_by_split.items()},
        "label_dist": label_dist,
        "minority_share": minority_share,
        "token_len": {
            "p50": _percentile(lengths, 0.5),
            "p95": _percentile(lengths, 0.95),
            "max": lengths[-1] if lengths else 0,
        },
        "n_over_max_len": n_over,
        "n_exact_dupes": n_exact,
        "n_cross_split_dupes": n_cross,
        "n_group_leak": n_leak,
        "unseen_labels": unseen,
        "warnings": warnings,
        "errors": errors,
    }
    if task == "token-cls":
        n_tags = sum(len(r["tags"]) for r in train)
        n_ent = sum(1 for r in train for t in r["tags"] if t != "O")
        report["entity_token_share"] = round(n_ent / n_tags, 4) if n_tags else 0.0
    return report


def val_size_warning(
    task: str, objective: str, params: dict, val_label_counts: dict[str, int], positive: str | None
) -> str | None:
    if objective in _RECALL_METRICS:
        recall = params.get("recall_min", params.get("recall"))
        if recall is None or positive is None:
            return None
        n_pos = val_label_counts.get(positive, 0)
        miss = 1 - recall
        if miss <= 0:
            return None
        if n_pos * miss < 10:
            need = math.ceil(10 / miss - 1e-9)
            return (
                f"val has {n_pos} positives; at recall {recall} the metric rests on "
                f"{n_pos * miss:.1f} missed positives (<10). Need about {need} val positives; "
                "use a larger val share."
            )
        return None
    if objective == "recall_at_precision":
        if positive is None:
            return None
        n_pos = val_label_counts.get(positive, 0)
        if n_pos < 100:
            return f"val has {n_pos} positives (<100) for recall_at_precision; use a larger val share."
        return None
    if not val_label_counts:
        return None
    name, n = min(val_label_counts.items(), key=lambda kv: kv[1])
    if n < 100:
        kind = "class" if task == "seq-cls" else "entity type"
        return f"smallest val {kind} {name!r} has {n} examples (<100); use a larger val share."
    return None


def _strat_key(row: dict) -> str:
    if "label" in row:
        return str(row["label"])
    return "entity" if any(t != "O" for t in row["tags"]) else "none"


def make_splits(rows, task, seed=0, frac=(0.8, 0.1, 0.1)):
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for r in rows:
        gid = r.get("group_id")
        groups.setdefault(f"g:{gid}" if gid is not None else f"i:{r['id']}", []).append(r)
    strata: dict[str, list[list[dict]]] = defaultdict(list)
    for members in groups.values():
        strata[Counter(_strat_key(r) for r in members).most_common(1)[0][0]].append(members)

    out: tuple[list, list, list] = ([], [], [])
    for key in sorted(strata):
        gs = strata[key]
        rng.shuffle(gs)
        total = sum(len(g) for g in gs)
        targets = [f * total for f in frac]
        got = [0, 0, 0]
        for g in sorted(gs, key=len, reverse=True):
            idx = max(
                (i for i in range(3) if targets[i] > 0),
                key=lambda i: (targets[i] - got[i]) / targets[i],
            )
            out[idx].extend(g)
            got[idx] += len(g)
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    blob = "".join(json.dumps(r) + "\n" for r in rows).encode()
    path.write_bytes(blob)
    return hashlib.sha256(blob).hexdigest()


def freeze_splits(exp_dir, uid, train, val, test) -> dict:
    splits = Path(exp_dir) / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    hashes = {
        "train": _write_jsonl(splits / "train.jsonl", train),
        "val": _write_jsonl(splits / "val.jsonl", val),
    }
    vault = vault_dir(uid)
    vault.mkdir(parents=True, exist_ok=True)
    vfile = vault / "test.jsonl"
    hashes["test"] = _write_jsonl(vfile, test)
    os.chmod(vfile, 0o600)
    (splits / "test.meta.json").write_text(
        json.dumps({"sha256": hashes["test"], "n_rows": len(test)})
    )
    return hashes


def read_test(uid: str) -> list[dict]:
    with (vault_dir(uid) / "test.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def subset(rows, n, seed):
    if n >= len(rows):
        return rows
    rng = random.Random(seed)
    strata: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        strata[_strat_key(r)].append(i)
    keys = sorted(strata)
    quota = {k: n * len(strata[k]) / len(rows) for k in keys}
    take = {k: int(quota[k]) for k in keys}
    for k in sorted(keys, key=lambda k: quota[k] - take[k], reverse=True)[: n - sum(take.values())]:
        take[k] += 1
    chosen: list[int] = []
    for k in keys:
        chosen.extend(rng.sample(strata[k], take[k]))
    return [rows[i] for i in sorted(chosen)]
