import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from finetune_kit import data, sources


def test_csv_with_bom_quotes_and_newlines(tmp_path):
    p = tmp_path / "train.csv"
    p.write_bytes('﻿text,label\n"a, b",0\n"line1\nline2",1\n'.encode("utf-8"))
    rows = data.load_rows(p, "seq-cls")
    assert [r["text"] for r in rows] == ["a, b", "line1\nline2"]
    assert [r["id"] for r in rows] == ["train-000001", "train-000002"]


def test_tsv_reads(tmp_path):
    p = tmp_path / "val.tsv"
    p.write_text("text\tlabel\nhello, there\t0\nbye\t1\n", encoding="utf-8")
    rows = data.load_rows(p, "seq-cls", role="val")
    assert [r["text"] for r in rows] == ["hello, there", "bye"]
    assert rows[0]["id"] == "val-000001"


def test_parquet_reads(tmp_path):
    p = tmp_path / "t.parquet"
    pq.write_table(pa.table({"tokens": [["a", "b"], ["c"]], "tags": [["O", "B-X"], ["O"]]}), p)
    rows = data.load_rows(p, "token-cls", role="test")
    assert rows[0] == {"id": "test-000001", "tokens": ["a", "b"], "tags": ["O", "B-X"]}
    assert rows[1]["id"] == "test-000002"


def test_column_mapping(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text("prompt,is_injection\nhi,0\nyo,1\n", encoding="utf-8")
    rows = data.load_rows(p, "seq-cls", columns={"text": "prompt", "label": "is_injection"})
    assert set(rows[0]) == {"id", "text", "label"}
    assert rows[0]["text"] == "hi" and rows[1]["label"] == "1"


def test_user_id_column_used_stringified(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text("pk,text,label\n7,hi,0\n8,yo,1\n", encoding="utf-8")
    rows = data.load_rows(p, "seq-cls", columns={"id": "pk"})
    assert [r["id"] for r in rows] == ["7", "8"]


def test_missing_mapped_column_lists_available(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text("prompt,is_injection\nhi,0\n", encoding="utf-8")
    with pytest.raises(ValueError) as e:
        data.load_rows(p, "seq-cls", columns={"text": "promt"})
    assert "no column 'promt'" in str(e.value)
    assert "['is_injection', 'prompt']" in str(e.value)


def test_empty_text_cell_errors_with_row(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text("text,label\nok,0\n,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="train.csv row 2: text must be a non-empty string"):
        data.load_rows(p, "seq-cls")


def test_empty_label_cell_refused(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text("text,label\nok,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="train.csv row 1: label must not be empty"):
        data.load_rows(p, "seq-cls")


def test_csv_token_cls_refused(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text("tokens,tags\na,O\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV/TSV support seq-cls only"):
        data.load_rows(p, "token-cls")


def test_unsupported_suffix(tmp_path):
    p = tmp_path / "x.xlsx"
    p.write_bytes(b"")
    with pytest.raises(ValueError) as e:
        sources.read_table(p)
    assert str(e.value) == "unsupported data file type .xlsx; use .jsonl, .csv, .tsv or .parquet"


def test_bad_parquet_is_value_error(tmp_path):
    p = tmp_path / "x.parquet"
    p.write_bytes(b"not parquet")
    with pytest.raises(ValueError, match="x.parquet"):
        sources.read_table(p)


def test_existing_jsonl_unchanged(tmp_path):
    p = tmp_path / "d.jsonl"
    rows = [{"id": "a", "text": "hi", "label": "0", "group_id": "g"}, {"id": "b", "text": "yo", "label": "1"}]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert data.load_rows(p, "seq-cls") == rows


def test_jsonl_errors_keep_line_numbers(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"id":"a","text":"hi","label":"0"}\n\n{"id":"b","text":" ","label":"1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"d.jsonl line 3: text must be a non-empty string"):
        data.load_rows(p, "seq-cls")


def test_normalize_class_label_names():
    raw = [{"text": "a", "label": 0}, {"text": "b", "label": 1}, {"text": "c", "label": -1}]
    rows = sources.normalize(raw, "seq-cls", {}, "train", "x", label_names={"label": ["neg", "pos"]})
    assert [r["label"] for r in rows] == ["neg", "pos", -1]


def test_normalize_identity_drops_extras():
    rows = sources.normalize([{"text": "a", "label": "0", "junk": 1}], "seq-cls", {}, "val", "x")
    assert rows == [{"id": "val-000001", "text": "a", "label": "0"}]
