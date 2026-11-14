"""Reading local data files (JSONL, CSV, TSV, Parquet) and mapping them to the kit's row schema."""
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from finetune_kit import data
from finetune_kit.data import DataError
from finetune_kit.paths import kit_home

_STANDARD = ("id", "text", "label", "tokens", "tags", "group_id")


def read_jsonl(path: Path) -> list[tuple[int, dict]]:
    """(line number, object) per non-blank line."""
    out = []
    with Path(path).open("rb") as f:
        for n, raw in enumerate(f, 1):
            where = f"{Path(path).name} line {n}"
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError as e:
                raise DataError(f"{where}: not valid UTF-8 ({e.reason})") from e
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise DataError(f"{where}: invalid JSON ({e.msg})") from e
            if not isinstance(row, dict):
                raise DataError(f"{where}: expected a JSON object")
            out.append((n, row))
    return out


def _read_delimited(path: Path, delimiter: str) -> list[dict]:
    csv.field_size_limit(2**31 - 1)
    kind = "TSV" if delimiter == "\t" else "CSV"
    try:
        with path.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f, delimiter=delimiter))
    except UnicodeDecodeError as e:
        raise ValueError(f"{path.name}: not valid UTF-8 ({e.reason})") from e
    except csv.Error as e:
        raise ValueError(f"{path.name}: invalid {kind} ({e})") from e


def _read_parquet(path: Path) -> list[dict]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        return pq.read_table(path).to_pylist()
    except (pa.ArrowException, OSError) as e:
        raise ValueError(f"{path.name}: cannot read Parquet ({' '.join(str(e).split())})") from e


def read_table(path: Path) -> list[dict]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [row for _, row in read_jsonl(path)]
    if suffix == ".csv":
        return _read_delimited(path, ",")
    if suffix == ".tsv":
        return _read_delimited(path, "\t")
    if suffix == ".parquet":
        return _read_parquet(path)
    raise ValueError(f"unsupported data file type {path.suffix}; use .jsonl, .csv, .tsv or .parquet")


def _names(value, names: list[str]):
    def one(v):
        if isinstance(v, int) and not isinstance(v, bool) and 0 <= v < len(names):
            return names[v]
        return v

    return [one(v) for v in value] if isinstance(value, list) else one(value)


def normalize(
    raw: list[dict], task: str, columns: dict[str, str], role: str, where: str,
    label_names: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Rename mapped columns to standard fields, drop the rest, add ids, apply ClassLabel names."""
    available: set[str] = set()
    for r in raw:
        available.update(k for k in r if isinstance(k, str))
    # Standard-named columns not remapped keep their name; other unmapped columns are dropped.
    mapping = {f: f for f in _STANDARD if f in available and f not in columns} | dict(columns)
    for col in mapping.values():
        if col not in available and raw:
            raise ValueError(f"{where}: no column {col!r}; columns are {sorted(available)}")
    label_names = label_names or {}
    rows = []
    for n, r in enumerate(raw, 1):
        row = {}
        for field, col in mapping.items():
            value = r.get(col)
            if col in label_names:
                value = _names(value, label_names[col])
            row[field] = value
        row_id = str(row.pop("id")) if "id" in row else f"{role}-{n:06d}"
        rows.append({"id": row_id, **row})
    return rows


# ---- Hugging Face ----

_SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("validation", "valid", "val", "dev"),
    "test": ("test",),
}


@dataclass
class Loaded:
    train: list[dict]
    val: list[dict] | None
    test: list[dict] | None
    spec: str
    ignored_splits: list[str]
    test_unlabelled: bool


def parse_hf(spec: str) -> tuple[str, str | None]:
    """`hf:<repo_id>` or `hf:<repo_id>:<config>`; repo_id may contain one `/`."""
    err = ValueError("Hugging Face source must be hf:<id> or hf:<id>:<config>")
    if not spec.startswith("hf:"):
        raise err
    parts = spec[3:].split(":")
    if len(parts) > 2 or not parts[0] or parts[0].count("/") > 1 or any(not p.strip() for p in parts):
        raise err
    if parts[0].startswith("/") or parts[0].endswith("/"):
        raise err
    return parts[0], parts[1] if len(parts) == 2 else None


def _spec(repo_id: str, config: str | None) -> str:
    return f"hf:{repo_id}" + (f":{config}" if config else "")


def _first_line(e: Exception) -> str:
    lines = str(e).strip().splitlines()
    return lines[0].strip() if lines else type(e).__name__


def _resolve_revision(repo_id: str, revision: str | None, config: str | None = None) -> str:
    try:
        from huggingface_hub import HfApi

        sha = HfApi().dataset_info(repo_id, revision=revision).sha
    except Exception as e:
        raise ValueError(f"could not load {_spec(repo_id, config)}: {_first_line(e)}") from e
    if not sha:
        raise ValueError(f"could not resolve a revision for {_spec(repo_id, config)}")
    return sha


def _load_dataset(repo_id: str, config: str | None, revision: str) -> dict[str, tuple[list[dict], dict]]:
    try:
        import datasets

        ds = datasets.load_dataset(
            repo_id, config, revision=revision, cache_dir=str(kit_home() / "hf-cache")
        )
        return {name: (ds[name].to_list(), dict(ds[name].features)) for name in ds}
    except Exception as e:
        raise ValueError(f"could not load {_spec(repo_id, config)}: {_first_line(e)}") from e


def pick_splits(names: list[str]) -> dict[str, str]:
    """role -> split name, matching aliases case-insensitively. Raises when there is no train split."""
    lower = {n.lower(): n for n in reversed(names)}
    picked = {}
    for role, aliases in _SPLIT_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                picked[role] = lower[alias]
                break
    if "train" not in picked:
        raise ValueError(f"no train split; splits are {names}")
    return picked


def _class_names(feature) -> list[str] | None:
    import datasets

    if isinstance(feature, datasets.ClassLabel):
        return list(feature.names)
    if isinstance(feature, list) and feature:
        return _class_names(feature[0])
    inner = getattr(feature, "feature", None)  # Sequence / List
    return _class_names(inner) if inner is not None else None


def label_names(features: dict, columns: dict[str, str], task: str) -> dict[str, list[str]]:
    """ClassLabel names for the source column mapped to label (seq-cls) or tags (token-cls)."""
    field = "label" if task == "seq-cls" else "tags"
    col = columns.get(field, field)
    names = _class_names(features[col]) if col in features else None
    return {col: names} if names is not None else {}


def load_hf(spec: str, revision: str, task: str, columns: dict) -> Loaded:
    """Normalized but unvalidated rows for the train/val/test splits found."""
    repo_id, config = parse_hf(spec)
    raw = _load_dataset(repo_id, config, revision)
    picked = pick_splits(list(raw))
    used = set(picked.values())
    out: dict[str, list[dict] | None] = {"train": None, "val": None, "test": None}
    for role, split in picked.items():
        rows, features = raw[split]
        out[role] = normalize(
            rows, task, columns, role, f"{spec} {split}", label_names(features, columns, task)
        )
    unlabelled = out["test"] is not None and _all_unlabelled(out["test"], task)
    return Loaded(
        train=out["train"],
        val=out["val"],
        test=out["test"],
        spec=spec,
        ignored_splits=[n for n in raw if n not in used],
        test_unlabelled=unlabelled,
    )


def _missing(v) -> bool:
    return v is None or v == "" or (isinstance(v, int) and not isinstance(v, bool) and v == -1)


def _all_unlabelled(rows: list[dict], task: str) -> bool:
    if task == "seq-cls":
        return all(_missing(r.get("label")) for r in rows)
    return all(
        not (r.get("tags") or []) or all(_missing(t) for t in r["tags"]) for r in rows
    )


def resolve_splits(loaded: Loaded, task: str, val_frac: float) -> tuple[list, list, list, dict]:
    """Return train, val, test and split_origin, carving missing parts from train, then validate."""
    train, val, test = loaded.train, loaded.val, loaded.test
    origin = {"train": "published", "val": "published", "test": "published"}
    val_missing = not val  # an empty published split counts as missing
    test_missing = not test
    carve_test = test_missing or loaded.test_unlabelled
    test_origin = (
        "carved from train (no published test split)"
        if test_missing
        else "carved from train (published test has no labels)"
    )
    if val_missing and carve_test:
        train, val, test = data.make_splits(train, task, frac=(1 - 0.1 - val_frac, val_frac, 0.1))
        origin["val"] = "carved from train"
        origin["test"] = test_origin
    elif val_missing:
        train, val, _ = data.make_splits(train, task, frac=(1 - val_frac, val_frac, 0))
        origin["val"] = "carved from train"
    elif carve_test:
        train, _, test = data.make_splits(train, task, frac=(0.9, 0, 0.1))
        origin["test"] = test_origin
    for role, rows in (("train", train), ("val", val), ("test", test)):
        data.validate_rows(rows, task, f"{loaded.spec} {role}")
    return train, val, test, origin


def _local_path(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else Path(root) / path


@dataclass
class Resolved:
    train: list[dict]
    val: list[dict]
    test: list[dict]
    origin: dict[str, str]
    ignored_splits: list[str]


def resolve_source(cfg_data, root: Path, task: str, val_frac: float) -> Resolved:
    """Train, val, test, split origin and ignored HF splits for the experiment's configured data."""
    cols = cfg_data.columns
    if cfg_data.source:
        if not cfg_data.revision:
            raise ValueError(f"{cfg_data.source} has no pinned revision in experiment.yaml")
        loaded = load_hf(cfg_data.source, cfg_data.revision, task, cols)
        return Resolved(*resolve_splits(loaded, task, val_frac), loaded.ignored_splits)
    train = data.load_rows(_local_path(root, cfg_data.train), task, cols, "train")
    if cfg_data.val and cfg_data.test:
        val = data.load_rows(_local_path(root, cfg_data.val), task, cols, "val")
        test = data.load_rows(_local_path(root, cfg_data.test), task, cols, "test")
        return Resolved(train, val, test, dict.fromkeys(("train", "val", "test"), "provided"), [])
    train, val, test = data.make_splits(train, task, frac=(1 - 0.1 - val_frac, val_frac, 0.1))
    origin = {"train": "provided", "val": "carved from train", "test": "carved from train"}
    return Resolved(train, val, test, origin, [])


def load_source(cfg_data, root: Path, task: str, val_frac: float) -> tuple[list, list, list, dict]:
    r = resolve_source(cfg_data, root, task, val_frac)
    return r.train, r.val, r.test, r.origin


def load_train(cfg_data, root: Path, task: str) -> list[dict]:
    """Train rows as the experiment will see them (named labels for HF ClassLabel columns)."""
    if cfg_data.source:
        return load_source(cfg_data, root, task, 0.1)[0]
    return data.load_rows(_local_path(root, cfg_data.train), task, cfg_data.columns, "train")


def pin_revision(spec: str, revision: str | None) -> str:
    """Commit sha for `spec` at `revision` (branch, tag, sha or None for the default branch)."""
    repo_id, config = parse_hf(spec)
    return _resolve_revision(repo_id, revision, config)


# ---- inspect-data: describe a dataset without showing any row text ----

_MAX_LABELS = 50
_SHOWN_LABELS = 20
_LABEL_NAMES = ("label", "labels", "class", "target")


def _value_type(v) -> str:
    if isinstance(v, bool):  # before int: bool is an int subclass
        return "bool"
    if isinstance(v, str):
        return "string"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, list) and v:
        inner = {_value_type(x) for x in v}
        if inner == {"string"}:
            return "list[string]"
        if inner == {"int"}:
            return "list[int]"
    return "other"


def _local_column_type(values: list) -> str:
    kinds = {_value_type(v) for v in values if v is not None}
    if len(kinds) == 1:
        return kinds.pop()
    return "float" if kinds == {"int", "float"} else "other"


def _feature_type(feature) -> str:
    import datasets

    if isinstance(feature, datasets.ClassLabel):
        return "classlabel"
    if isinstance(feature, datasets.Value):
        dtype = feature.dtype
        if dtype in ("string", "large_string"):
            return "string"
        if dtype.startswith(("int", "uint")):
            return "int"
        if dtype.startswith("float"):
            return "float"
        if dtype == "bool":
            return "bool"
        return "other"
    inner = feature[0] if isinstance(feature, list) and feature else getattr(feature, "feature", None)
    if inner is not None:
        kind = _feature_type(inner)
        if kind in ("string", "int", "classlabel"):
            return f"list[{kind}]"
    return "other"


def _label_counts(rows: list[dict], col: str, names: list[str] | None) -> tuple[int, dict]:
    """(n_distinct, top values by count) for a column, or (n, {}) when it has too many values."""
    counts: dict = {}
    for r in rows:
        v = r.get(col)
        if v is None:
            continue
        v = _names(v, names) if names else v
        key = v if isinstance(v, str) else str(v)
        counts[key] = counts.get(key, 0) + 1
        if len(counts) > _MAX_LABELS:
            return len(counts), {}
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:_SHOWN_LABELS]
    return len(counts), dict(top)


def _text_column(rows: list[dict], columns: dict[str, str]) -> str | None:
    """The string column with the longest mean length."""
    def mean_len(col):
        lens = [len(r[col]) for r in rows if isinstance(r.get(col), str)]
        return sum(lens) / len(lens) if lens else 0

    strings = [c for c, t in columns.items() if t == "string"]
    return max(strings, key=mean_len) if strings else None


def _suggest(text: str | None, columns: dict[str, str], candidates: dict) -> tuple[dict, list[str]]:
    suggested: dict[str, str] = {}
    lists = [c for c, t in columns.items() if t == "list[string]"]
    tag_cols = [c for c, t in columns.items() if t in ("list[classlabel]", "list[string]")]
    tasks = []
    if text:
        suggested["text"] = text
        if candidates:
            named = [c for c in candidates if c.lower() in _LABEL_NAMES]
            suggested["label"] = named[0] if named else min(candidates, key=lambda c: candidates[c]["n_distinct"])
            tasks.append("seq-cls")
    if lists:
        tokens = lists[0]
        suggested["tokens"] = tokens
        tags = [c for c in tag_cols if c != tokens]
        if tags:
            suggested["tags"] = tags[0]
            tasks.append("token-cls")
    return suggested, tasks


def inspect(
    source: str | None, train: str | None, val: str | None, test: str | None, root: Path,
    revision: str | None = None,
) -> dict:
    """Describe a dataset's splits, columns, label counts and a suggested column mapping, never its text."""
    from finetune_kit.experiment import check_data_choice

    check_data_choice(train, source, val, test)
    if revision is not None and not source:
        raise ValueError("--revision only applies with --source")
    splits: dict[str, dict] = {}
    names_by_col: dict[str, list[str]] = {}
    if source:
        repo_id, config = parse_hf(source)
        sha = _resolve_revision(repo_id, revision, config)
        raw = _load_dataset(repo_id, config, sha)
        picked = pick_splits(list(raw))
        role_of = {name: role for role, name in picked.items()}
        for name, (rows, _) in raw.items():
            splits[name] = {"n_rows": len(rows), "role": role_of.get(name)}
        train_rows, features = raw[picked["train"]]
        columns = {c: _feature_type(f) for c, f in features.items()}
        for c, f in features.items():
            names = _class_names(f)
            if names is not None:
                names_by_col[c] = names
        shown_source = source
    else:
        sha = None
        train_rows = read_table(_local_path(root, train))
        splits["train"] = {"n_rows": len(train_rows), "role": "train"}
        for role, p in (("val", val), ("test", test)):
            if p:
                splits[role] = {"n_rows": len(read_table(_local_path(root, p))), "role": role}
        cols: dict[str, list] = {}
        for r in train_rows:
            for c in r:
                cols.setdefault(c, [])
        for c, vals in cols.items():
            vals.extend(r.get(c) for r in train_rows)
        columns = {c: _local_column_type(v) for c, v in cols.items()}
        shown_source = train
    text = _text_column(train_rows, columns)
    candidates = {}
    for c, t in columns.items():
        if c == text:
            continue  # the text column is never shown, however few distinct values it has
        if t in ("int", "bool", "classlabel", "string"):
            n, values = _label_counts(train_rows, c, names_by_col.get(c))
            # A string column whose values mostly differ per row is text, not labels: never print it.
            text_like = t == "string" and n * 2 > sum(r.get(c) is not None for r in train_rows)
            if n <= _MAX_LABELS and not text_like:
                candidates[c] = {"n_distinct": n, "values": values}
    suggested, tasks = _suggest(text, columns, candidates)
    return {
        "source": shown_source,
        "revision": sha,
        "splits": splits,
        "columns": columns,
        "label_candidates": candidates,
        "suggested_columns": suggested,
        "tasks_possible": tasks,
    }
