"""Reading local data files (JSONL, CSV, TSV, Parquet) and mapping them to the kit's row schema."""
import csv
import json
from pathlib import Path

from finetune_kit.data import DataError

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
