import json
from pathlib import Path

import datasets
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


# ---- Hugging Face loading and split resolution ----

_CL = datasets.ClassLabel(names=["benign", "injection"])


def _seq_rows(n, label=lambda i: i % 2):
    return [{"text": f"t{i}", "label": label(i), "extra": i} for i in range(n)]


def test_parse_hf():
    assert sources.parse_hf("hf:glue") == ("glue", None)
    assert sources.parse_hf("hf:glue:sst2") == ("glue", "sst2")
    assert sources.parse_hf("hf:a/b:cfg") == ("a/b", "cfg")
    with pytest.raises(ValueError, match="Hugging Face source must be"):
        sources.parse_hf("hf:a/b/c")


def test_pick_splits_aliases():
    assert sources.pick_splits(["train", "dev", "test"]) == {"train": "train", "val": "dev", "test": "test"}
    assert sources.pick_splits(["TRAIN", "Validation"]) == {"train": "TRAIN", "val": "Validation"}
    with pytest.raises(ValueError, match="no train split"):
        sources.pick_splits(["foo"])


def test_classlabel_names_applied(hf_fake):
    hf_fake({"train": _seq_rows(4)})
    loaded = sources.load_hf("hf:x/y", "abc123", "seq-cls", {})
    assert [r["label"] for r in loaded.train] == ["benign", "injection", "benign", "injection"]
    assert "extra" not in loaded.train[0]


def test_token_cls_names():
    feats = {"tags": datasets.Sequence(datasets.ClassLabel(names=["O", "B-X"]))}
    assert sources.label_names(feats, {}, "token-cls") == {"tags": ["O", "B-X"]}
    assert sources.label_names({"y": _CL}, {"label": "y"}, "seq-cls") == {"y": ["benign", "injection"]}
    assert sources.label_names({"label": datasets.Value("string")}, {}, "seq-cls") == {}


def test_all_published_used_as_is(hf_fake):
    hf_fake({"train": _seq_rows(8), "validation": _seq_rows(2), "test": _seq_rows(2)})
    loaded = sources.load_hf("hf:x/y", "abc123", "seq-cls", {})
    tr, va, te, origin = sources.resolve_splits(loaded, "seq-cls", 0.1)
    assert (len(tr), len(va), len(te)) == (8, 2, 2)
    assert origin == {"train": "published", "val": "published", "test": "published"}


def test_unlabelled_test_carved(hf_fake):
    hf_fake({"train": _seq_rows(40), "validation": _seq_rows(4), "test": _seq_rows(4, lambda i: -1)})
    loaded = sources.load_hf("hf:x/y", "abc123", "seq-cls", {})
    assert loaded.test_unlabelled
    tr, va, te, origin = sources.resolve_splits(loaded, "seq-cls", 0.1)
    assert origin["test"] == "carved from train (published test has no labels)"
    assert origin["val"] == "published"
    assert len(te) == 4 and len(tr) == 36 and all(r["label"] in ("benign", "injection") for r in te)


def test_missing_val_carved(hf_fake):
    hf_fake({"train": _seq_rows(40), "test": _seq_rows(6)})
    tr, va, te, origin = sources.resolve_splits(
        sources.load_hf("hf:x/y", "abc123", "seq-cls", {}), "seq-cls", 0.1
    )
    assert origin == {"train": "published", "val": "carved from train", "test": "published"}
    assert len(va) == 4 and len(tr) == 36 and len(te) == 6


def test_only_train_carved(hf_fake):
    hf_fake({"train": _seq_rows(100)})
    tr, va, te, origin = sources.resolve_splits(
        sources.load_hf("hf:x/y", "abc123", "seq-cls", {}), "seq-cls", 0.1
    )
    assert len(tr) + len(va) + len(te) == 100 and va and te
    assert origin["val"] == "carved from train"
    assert origin["test"] == "carved from train (no published test split)"


def test_generated_ids_unique_across_splits(hf_fake):
    hf_fake({"train": _seq_rows(8), "validation": _seq_rows(2), "test": _seq_rows(2)})
    tr, va, te, _ = sources.resolve_splits(
        sources.load_hf("hf:x/y", "abc123", "seq-cls", {}), "seq-cls", 0.1
    )
    ids = [r["id"] for r in tr + va + te]
    assert len(ids) == len(set(ids)) == 12


def test_ignored_splits_reported(hf_fake):
    hf_fake({"train": _seq_rows(4), "unsupervised": _seq_rows(3)})
    assert sources.load_hf("hf:x/y", "abc123", "seq-cls", {}).ignored_splits == ["unsupervised"]


def test_library_error_is_one_line_value_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("Dataset 'x/y' doesn't exist\nTraceback noise")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id, **kw: "/nonexistent")
    monkeypatch.setattr(datasets, "load_dataset", boom)
    with pytest.raises(ValueError) as e:
        sources._load_dataset("x/y", "cfg", "abc")
    assert str(e.value) == "could not load hf:x/y:cfg: Dataset 'x/y' doesn't exist"


def test_empty_published_val_and_test_count_as_missing(hf_fake):
    hf_fake({"train": _seq_rows(100), "validation": [], "test": []})
    tr, va, te, origin = sources.resolve_splits(
        sources.load_hf("hf:x/y", "abc123", "seq-cls", {}), "seq-cls", 0.1
    )
    assert va and te and len(tr) + len(va) + len(te) == 100
    assert origin["val"] == "carved from train"
    assert origin["test"] == "carved from train (no published test split)"


def test_validation_errors_name_the_dataset(hf_fake):
    rows = _seq_rows(8)
    rows[0]["text"] = " "
    hf_fake({"train": rows, "validation": _seq_rows(2), "test": _seq_rows(2)})
    loaded = sources.load_hf("hf:x/y:cfg", "abc123", "seq-cls", {})
    with pytest.raises(ValueError, match="hf:x/y:cfg train"):
        sources.resolve_splits(loaded, "seq-cls", 0.1)


def test_token_cls_classlabel_tags_end_to_end(hf_fake):
    names = ["O", "B-PER", "I-PER"]
    feats = {"tokens": datasets.Sequence(datasets.Value("string")),
             "tags": datasets.Sequence(datasets.ClassLabel(names=names))}

    def rows(n):
        return [{"tokens": ["a", "b", "c"], "tags": [0, 1, 2]} for _ in range(n)]

    hf_fake({"train": rows(8), "validation": rows(2), "test": rows(2)}, features=feats)
    loaded = sources.load_hf("hf:x/y", "abc123", "token-cls", {})
    tr, va, te, origin = sources.resolve_splits(loaded, "token-cls", 0.1)
    assert tr[0]["tags"] == ["O", "B-PER", "I-PER"]
    assert data.label_map(tr, "token-cls") == ["B-PER", "I-PER", "O"]


def test_resolve_revision_falsy_sha_and_config_in_error(monkeypatch):
    import huggingface_hub

    class Api:
        def dataset_info(self, repo, revision=None):
            return type("I", (), {"sha": None})()

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    with pytest.raises(ValueError, match="could not resolve a revision for hf:x/y:cfg"):
        sources._resolve_revision("x/y", None, "cfg")


def test_load_source_local_origins(tmp_path):
    from finetune_kit.experiment import DataPaths

    rows = [{"id": f"s{i}", "text": f"t{i}", "label": str(i % 2)} for i in range(40)]
    (tmp_path / "d.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    tr, va, te, origin = sources.load_source(DataPaths(train="d.jsonl"), tmp_path, "seq-cls", 0.1)
    assert origin == {"train": "provided", "val": "carved from train", "test": "carved from train"}
    assert len(tr) + len(va) + len(te) == 40


# ---- inspect ----


def test_inspect_csv_suggests_mapping(tmp_path):
    (tmp_path / "t.csv").write_text(
        "prompt,is_injection\n" + "".join(f"prompt number {i},{i % 2}\n" for i in range(6)), encoding="utf-8"
    )
    r = sources.inspect(None, "t.csv", None, None, tmp_path)
    assert r["source"] == "t.csv" and r["revision"] is None
    assert r["splits"] == {"train": {"n_rows": 6, "role": "train"}}
    assert r["columns"] == {"prompt": "string", "is_injection": "string"}
    assert list(r["label_candidates"]) == ["is_injection"]  # unique prompts are not label values
    assert r["label_candidates"]["is_injection"] == {"n_distinct": 2, "values": {"0": 3, "1": 3}}
    assert r["suggested_columns"] == {"text": "prompt", "label": "is_injection"}
    assert r["tasks_possible"] == ["seq-cls"]


def test_inspect_hf_shows_classlabel_names(hf_fake):
    rows = [{"text": f"t{i}", "label": 1 if i < 3 else 0} for i in range(10)]
    feats = {"text": datasets.Value("string"), "label": _CL}
    hf_fake({"train": rows, "dev": rows[:2], "extra": rows[:1]}, features=feats)
    r = sources.inspect("hf:x/y", None, None, None, Path("."), revision="main")
    assert r["source"] == "hf:x/y" and r["revision"] == "abc123"
    assert r["splits"] == {
        "train": {"n_rows": 10, "role": "train"},
        "dev": {"n_rows": 2, "role": "val"},
        "extra": {"n_rows": 1, "role": None},
    }
    assert r["columns"] == {"text": "string", "label": "classlabel"}
    assert r["label_candidates"]["label"] == {"n_distinct": 2, "values": {"benign": 7, "injection": 3}}
    assert r["suggested_columns"] == {"text": "text", "label": "label"}


def test_inspect_never_prints_text(tmp_path):
    secret = "SENTINEL-zq81-secret"
    rows = [{"text": f"{secret} {i % 3}", "label": i % 2, "n": i} for i in range(6)]  # repeated texts too
    (tmp_path / "t.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = json.dumps(sources.inspect(None, "t.jsonl", None, None, tmp_path))
    assert secret not in out


def test_inspect_token_cls_and_label_limits(hf_fake):
    names = ["O", "B-PER"]
    feats = {"tokens": datasets.Sequence(datasets.Value("string")),
             "tags": datasets.Sequence(datasets.ClassLabel(names=names)),
             "id": datasets.Value("int64")}
    rows = [{"tokens": ["a", "b"], "tags": [0, 1], "id": i} for i in range(60)]
    hf_fake({"train": rows}, features=feats)
    r = sources.inspect("hf:x/y", None, None, None, Path("."))
    assert r["columns"] == {"tokens": "list[string]", "tags": "list[classlabel]", "id": "int"}
    assert "id" not in r["label_candidates"]  # 60 distinct > 50
    assert r["suggested_columns"] == {"tokens": "tokens", "tags": "tags"}
    assert r["tasks_possible"] == ["token-cls"]


def test_inspect_values_capped_at_20_and_types(tmp_path):
    rows = [{"text": "x", "label": i % 30, "flag": True, "f": 1.5, "o": {"a": 1}} for i in range(60)]
    (tmp_path / "t.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    r = sources.inspect(None, "t.jsonl", None, None, tmp_path)
    assert r["columns"] == {"text": "string", "label": "int", "flag": "bool", "f": "float", "o": "other"}
    assert r["label_candidates"]["label"]["n_distinct"] == 30
    assert len(r["label_candidates"]["label"]["values"]) == 20
    assert r["suggested_columns"]["label"] == "label"


def test_inspect_choice_errors(tmp_path):
    with pytest.raises(ValueError, match="give --train or --source, not both"):
        sources.inspect("hf:x/y", "a.csv", None, None, tmp_path)
    with pytest.raises(ValueError, match="give --train or --source"):
        sources.inspect(None, None, None, None, tmp_path)


def test_inspect_never_prints_second_text_column(tmp_path):
    long_rep = "SENTINEL-REPEATED-" + "x" * 40  # > 40 chars, low cardinality
    rows = [
        {"text": f"a fairly long sentence number {i} with plenty of extra words so it is clearly the longest column here", "label": "spam" if i % 2 else "ham",
         "title": f"SENTINEL-UNIQUE-{i}", "ctx": long_rep, "short": f"S-{i % 3}"}
        for i in range(12)
    ]
    (tmp_path / "t.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    r = sources.inspect(None, "t.jsonl", None, None, tmp_path)
    out = json.dumps(r)
    assert "SENTINEL" not in out
    assert "title" not in r["label_candidates"] and "ctx" not in r["label_candidates"]
    assert r["label_candidates"]["label"]["values"] == {"ham": 6, "spam": 6}
    assert r["suggested_columns"]["label"] == "label"
    assert r["suggested_columns"]["text"] == "text"  # so ctx reaches the 40-character guard


def test_inspect_id_column_is_not_text(hf_fake):
    feats = {"id": datasets.Value("string"), "tokens": datasets.Sequence(datasets.Value("string")),
             "tags": datasets.Sequence(datasets.ClassLabel(names=["O", "B"]))}
    rows = [{"id": f"doc-{i}-long-identifier", "tokens": ["a"], "tags": [0]} for i in range(5)]
    hf_fake({"train": rows}, features=feats)
    r = sources.inspect("hf:x/y", None, None, None, Path("."))
    assert "text" not in r["suggested_columns"]
    assert r["tasks_possible"] == ["token-cls"]
    assert "id" not in r["label_candidates"] or r["label_candidates"]["id"]["n_distinct"] <= 2


# ---- final-review fixes ----

def _mapped_rows(n, label=lambda i: i % 2):
    return [{"prompt": f"p{i}", "is_inj": label(i)} for i in range(n)]


_MAP = {"text": "prompt", "label": "is_inj"}


def test_mapped_label_missing_from_test_split_carves(hf_fake):
    test = [{"prompt": f"q{i}"} for i in range(4)]
    hf_fake({"train": _mapped_rows(40), "validation": _mapped_rows(4), "test": test},
            features={"is_inj": _CL})
    loaded = sources.load_hf("hf:a/b", "abc123", "seq-cls", _MAP)
    assert loaded.test_unlabelled
    tr, va, te, origin = sources.resolve_splits(loaded, "seq-cls", 0.1)
    assert origin["test"] == "carved from train (published test has no labels)"
    assert origin["val"] == "published" and len(te) == 4 and len(tr) == 36


def test_val_without_label_column_carved(hf_fake):
    val = [{"prompt": f"v{i}"} for i in range(4)]
    hf_fake({"train": _mapped_rows(40), "validation": val, "test": _mapped_rows(4)},
            features={"is_inj": _CL})
    loaded = sources.load_hf("hf:a/b", "abc123", "seq-cls", _MAP)
    tr, va, te, origin = sources.resolve_splits(loaded, "seq-cls", 0.1)
    assert origin["val"] == "carved from train (published validation has no labels)"
    assert origin["test"] == "published"
    assert len(va) == 4 and len(tr) == 36 and all(r["label"] in ("benign", "injection") for r in va)


def test_unlabelled_val_and_missing_test_both_carved(hf_fake):
    hf_fake({"train": _seq_rows(100), "validation": _seq_rows(4, lambda i: -1)})
    tr, va, te, origin = sources.resolve_splits(
        sources.load_hf("hf:x/y", "abc123", "seq-cls", {}), "seq-cls", 0.1
    )
    assert origin["val"] == "carved from train (published validation has no labels)"
    assert origin["test"] == "carved from train (no published test split)"
    assert len(tr) + len(va) + len(te) == 100


def test_partially_unlabelled_test_refused(hf_fake):
    hf_fake({"train": _seq_rows(40), "validation": _seq_rows(4),
             "test": _seq_rows(5, lambda i: -1 if i < 2 else i % 2)})
    loaded = sources.load_hf("hf:x/y", "abc123", "seq-cls", {})
    with pytest.raises(ValueError) as e:
        sources.resolve_splits(loaded, "seq-cls", 0.1)
    assert str(e.value) == (
        "hf:x/y test split has 2 rows without labels (of 5); drop them from the dataset or use a split with labels"
    )


def test_id_null_generates_ids(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text("id,text,label\n7,hi,0\n7,yo,1\n", encoding="utf-8")
    rows = data.load_rows(p, "seq-cls", columns={"id": None})
    assert [r["id"] for r in rows] == ["train-000001", "train-000002"]


def test_empty_or_null_id_refused(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text("id,text,label\n7,hi,0\n,yo,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"^train.csv row 2: id is empty$"):
        data.load_rows(p, "seq-cls")
    rows = [{"pk": "a", "text": "x", "label": 0}, {"pk": None, "text": "y", "label": 1}]
    with pytest.raises(ValueError, match=r"^hf:x/y train row 2: id is empty$"):
        sources.normalize(rows, "seq-cls", {"id": "pk"}, "train", "hf:x/y train")
    j = tmp_path / "t.jsonl"
    j.write_text('{"id": "a", "text": "x", "label": 0}\n{"id": null, "text": "y", "label": 1}\n')
    with pytest.raises(ValueError, match="t.jsonl line 2: id is empty"):
        data.load_rows(j, "seq-cls")


def test_only_id_may_be_null(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text("text,label\nhi,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="only id may be null"):
        data.load_rows(p, "seq-cls", columns={"label": None})


def test_label_column_looks_like_text():
    secret = "SENTINEL-text-value-that-is-quite-long-indeed-0"
    rows = [{"id": "a", "text": "t", "label": secret}, {"id": "b", "text": "t", "label": "x"}]
    with pytest.raises(ValueError) as e:
        data.check_label_column(rows)
    assert secret not in str(e.value)
    assert "is --columns mapping the text column as the label?" in str(e.value)
    many = [{"id": str(i), "text": "t", "label": f"l{i}"} for i in range(201)]
    with pytest.raises(ValueError, match="label column has 201 distinct values"):
        data.check_label_column(many)
    data.check_label_column(many[:200])  # 200 short labels are fine


def test_inspect_ragged_csv_skips_none_key(tmp_path):
    (tmp_path / "t.csv").write_text(
        "text,label\n" + "".join(f"some text {i},{i % 2},extra,more\n" for i in range(6)), encoding="utf-8"
    )
    r = sources.inspect(None, "t.csv", None, None, tmp_path)
    assert set(r["columns"]) == {"text", "label"}
    assert "tokens" not in r["suggested_columns"]
    json.dumps(r)


def test_load_dataset_uses_kit_cache_snapshot(monkeypatch, tmp_path):
    import huggingface_hub

    from finetune_kit.paths import kit_home

    calls = {}
    snap = tmp_path / "snap"

    def fake_snapshot(repo_id, **kw):
        calls["snapshot"] = (repo_id, kw)
        return str(snap)

    class _Split:
        features = {"label": _CL}

        def to_list(self):
            return [{"text": "a", "label": 0}]

    def fake_load(path, name=None, **kw):
        calls["load"] = (path, name, kw)
        return {"train": _Split()}

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    out = sources._load_dataset("a/b", "cfg", "deadbeef")
    cache = str(kit_home() / "hf-cache")
    assert calls["snapshot"] == ("a/b", {"repo_type": "dataset", "revision": "deadbeef", "cache_dir": cache})
    assert calls["load"] == (str(snap), "cfg", {"cache_dir": cache})
    assert out == {"train": ([{"text": "a", "label": 0}], {"label": _CL})}


def test_load_dataset_retries_offline(monkeypatch, tmp_path):
    import huggingface_hub

    seen = []

    def fake_snapshot(repo_id, **kw):
        seen.append(kw.get("local_files_only", False))
        if not kw.get("local_files_only"):
            raise OSError("connection refused")
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(datasets, "load_dataset", lambda path, name=None, **kw: {})
    assert sources._load_dataset("a/b", None, "deadbeef") == {}
    assert seen == [False, True]


def test_load_dataset_offline_failure_reports_original_error(monkeypatch):
    import huggingface_hub

    def fake_snapshot(repo_id, **kw):
        raise OSError("not cached" if kw.get("local_files_only") else "connection refused\nmore")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    with pytest.raises(ValueError) as e:
        sources._load_dataset("a/b", None, "deadbeef")
    assert str(e.value) == "could not load hf:a/b: connection refused"
