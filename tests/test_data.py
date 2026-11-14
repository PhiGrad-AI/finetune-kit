import json
import stat

import pytest

from finetune_kit import data
from finetune_kit.data import DataError
from finetune_kit.paths import vault_dir


class Tok:
    def __init__(self, k=5):
        self.k = k

    def __call__(self, texts, add_special_tokens=True):
        return {"input_ids": [[0] * (self.k if not callable(self.k) else self.k(t)) for t in texts]}


def seq_rows(n, pos_frac=0.3, groups=None):
    rows = []
    for i in range(n):
        r = {"id": f"r{i}", "text": f"text {i}", "label": "pos" if i % 10 < pos_frac * 10 else "neg"}
        if groups:
            r["group_id"] = f"g{i % groups}"
        rows.append(r)
    return rows


def test_bad_json_line_names_line(seq_jsonl):
    p = seq_jsonl([{"id": "a", "text": "x", "label": 0}])
    with p.open("a") as f:
        f.write("{not json\n")
    with pytest.raises(DataError, match="line 2"):
        data.load_rows(p, "seq-cls")


def test_token_len_mismatch(tok_jsonl):
    p = tok_jsonl([{"id": "a", "tokens": ["a", "b"], "tags": ["O"]}])
    with pytest.raises(DataError, match="tokens/tags length"):
        data.load_rows(p, "token-cls")


def test_bad_tag_and_missing_field_and_dupe_id(seq_jsonl, tok_jsonl):
    p = tok_jsonl([{"id": "a", "tokens": ["a"], "tags": ["X-PER"]}])
    with pytest.raises(DataError, match="tag"):
        data.load_rows(p, "token-cls")
    p = seq_jsonl([{"id": "a", "text": "x"}])
    with pytest.raises(DataError, match="label"):
        data.load_rows(p, "seq-cls")
    p = seq_jsonl([{"id": "a", "text": "x", "label": 1}, {"id": "a", "text": "y", "label": 0}])
    with pytest.raises(DataError, match="duplicate id"):
        data.load_rows(p, "seq-cls")


def test_string_labels_mapped(seq_jsonl):
    p = seq_jsonl([{"id": "1", "text": "a", "label": "pii"}, {"id": "2", "text": "b", "label": "clean"}])
    rows = data.load_rows(p, "seq-cls")
    assert data.label_map(rows, "seq-cls") == ["clean", "pii"]


def test_int_labels_stringified_and_token_labels(tok_jsonl, seq_jsonl):
    rows = data.load_rows(seq_jsonl([{"id": "1", "text": "a", "label": 1}, {"id": "2", "text": "b", "label": 0}]), "seq-cls")
    assert data.label_map(rows, "seq-cls") == ["0", "1"]
    trows = data.load_rows(
        tok_jsonl([{"id": "1", "tokens": ["a", "b"], "tags": ["B-PER", "I-PER"]}, {"id": "2", "tokens": ["c"], "tags": ["O"]}]),
        "token-cls",
    )
    assert data.label_map(trows, "token-cls") == ["B-PER", "I-PER", "O"]


def test_unseen_label_reported(tmp_path):
    train = [{"id": "1", "text": "a", "label": "a"}]
    val = [{"id": "2", "text": "b", "label": "x"}]
    rep = data.check(tmp_path, "seq-cls", 512, Tok(), {"train": train, "val": val})
    assert rep["unseen_labels"] == ["x"]
    assert rep["n_rows"] == {"train": 1, "val": 1}
    assert any("x" in w for w in rep["warnings"])


def test_group_split_no_leak():
    rows = seq_rows(100, groups=20)
    parts = data.make_splits(rows, "seq-cls", seed=1)
    seen = [{r["group_id"] for r in p} for p in parts]
    assert not (seen[0] & seen[1]) and not (seen[0] & seen[2]) and not (seen[1] & seen[2])
    assert sum(len(p) for p in parts) == 100


def test_split_stratified():
    rows = [{"id": str(i), "text": f"t{i}", "label": "pos" if i % 10 < 3 else "neg"} for i in range(1000)]
    for part in data.make_splits(rows, "seq-cls", seed=3):
        share = sum(r["label"] == "pos" for r in part) / len(part)
        assert abs(share - 0.3) <= 0.05


def test_split_deterministic_and_custom_frac():
    rows = seq_rows(200)
    a = data.make_splits(rows, "seq-cls", seed=2, frac=(0.6, 0.3, 0.1))
    b = data.make_splits(rows, "seq-cls", seed=2, frac=(0.6, 0.3, 0.1))
    assert a == b
    assert abs(len(a[1]) - 60) <= 3


def test_token_split_stratified_by_entity_presence():
    rows = [
        {"id": str(i), "tokens": ["a"], "tags": ["B-X"] if i % 4 == 0 else ["O"]} for i in range(400)
    ]
    for part in data.make_splits(rows, "token-cls", seed=0):
        share = sum(t != "O" for r in part for t in r["tags"]) / len(part)
        assert abs(share - 0.25) <= 0.05


def test_freeze_moves_test_to_vault(tmp_path):
    exp = tmp_path / "exp"
    rows = seq_rows(50)
    train, val, test = rows[:30], rows[30:40], rows[40:]
    hashes = data.freeze_splits(exp, "uid1", train, val, test)
    assert (exp / "splits" / "train.jsonl").exists()
    assert (exp / "splits" / "val.jsonl").exists()
    assert not (exp / "splits" / "test.jsonl").exists()
    vf = vault_dir("uid1") / "test.jsonl"
    assert stat.S_IMODE(vf.stat().st_mode) == 0o600
    meta = json.loads((exp / "splits" / "test.meta.json").read_text())
    assert meta["n_rows"] == 10 and meta["sha256"] == hashes["test"]
    assert set(hashes) == {"train", "val", "test"}
    assert data.read_test("uid1") == test


def test_cross_split_dupe_counted(tmp_path):
    train = [{"id": "1", "text": "Hello  World", "label": "a"}]
    val = [{"id": "2", "text": "hello world", "label": "a"}]
    rep = data.check(tmp_path, "seq-cls", 512, Tok(), {"train": train, "val": val})
    assert rep["n_cross_split_dupes"] == 1
    assert rep["n_exact_dupes"] == 0


def test_exact_dupes_and_group_leak(tmp_path):
    train = [
        {"id": "1", "text": "same", "label": "a", "group_id": "g"},
        {"id": "2", "text": "same", "label": "b", "group_id": "h"},
    ]
    val = [{"id": "3", "text": "other", "label": "a", "group_id": "g"}]
    rep = data.check(tmp_path, "seq-cls", 512, Tok(), {"train": train, "val": val})
    assert rep["n_exact_dupes"] == 1
    assert rep["n_group_leak"] == 1


def test_over_max_len_counted(tmp_path):
    train = [{"id": "1", "text": "long", "label": "a"}, {"id": "2", "text": "short", "label": "a"}]
    tok = Tok(lambda t: 600 if t == "long" else 10)
    rep = data.check(tmp_path, "seq-cls", 512, tok, {"train": train})
    assert rep["n_over_max_len"] == 1
    assert rep["token_len"]["max"] == 600
    assert any("512" in w for w in rep["warnings"])


def test_token_cls_check(tmp_path):
    train = [
        {"id": "1", "tokens": ["a", "b"], "tags": ["B-PER", "O"]},
        {"id": "2", "tokens": ["c", "d"], "tags": ["O", "O"]},
    ]
    seen = []

    def tok(texts, add_special_tokens=True):
        seen.extend(texts)
        return {"input_ids": [[0] * 3 for _ in texts]}

    rep = data.check(tmp_path, "token-cls", 512, tok, {"train": train})
    assert seen == ["a b", "c d"]
    assert rep["entity_token_share"] == 0.25
    assert rep["errors"] == []


def test_check_errors_on_missing_train(tmp_path):
    rep = data.check(tmp_path, "seq-cls", 512, Tok(), {"train": [], "val": []})
    assert rep["errors"]


def test_val_warning_pauc():
    w = data.val_size_warning("seq-cls", "pauc_hi", {"recall_min": 0.98}, {"pii": 266, "clean": 5000}, "pii")
    assert w and "500" in w and "val" in w


def test_val_warning_none_when_enough():
    assert data.val_size_warning("seq-cls", "pauc_hi", {"recall_min": 0.98}, {"pii": 600, "clean": 5000}, "pii") is None


def test_val_warning_recall_at_precision():
    assert data.val_size_warning("seq-cls", "recall_at_precision", {"precision": 0.9}, {"p": 50, "n": 900}, "p")
    assert data.val_size_warning("seq-cls", "recall_at_precision", {"precision": 0.9}, {"p": 150, "n": 900}, "p") is None


def test_val_warning_small_class_f1():
    assert data.val_size_warning("seq-cls", "macro_f1", {}, {"a": 40, "b": 500}, None)
    assert data.val_size_warning("seq-cls", "macro_f1", {}, {"a": 400, "b": 500}, None) is None


def test_subset_stratified_and_noop():
    rows = [{"id": str(i), "text": "t", "label": "pos" if i % 10 == 0 else "neg"} for i in range(1000)]
    sub = data.subset(rows, 100, seed=0)
    assert len(sub) == 100
    assert sum(r["label"] == "pos" for r in sub) == 10
    assert data.subset(rows, 5000, seed=0) == rows
    assert data.subset(rows, 100, seed=0) == sub


@pytest.mark.parametrize("row, msg", [
    ({"id": 1, "text": "", "label": "a"}, "text must be a non-empty string"),
    ({"id": 1, "text": 5, "label": "a"}, "text must be a non-empty string"),
    ({"id": 1, "text": "hi", "label": None}, "label must not be null"),
])
def test_load_rows_rejects_bad_seq_fields(tmp_path, row, msg):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"id": 0, "text": "ok", "label": "a"}) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(DataError, match=f"d.jsonl line 2: {msg}"):
        data.load_rows(p, "seq-cls")


def test_load_rows_rejects_non_string_tokens(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"id": 0, "tokens": ["a", 3], "tags": ["O", "O"]}) + "\n")
    with pytest.raises(DataError, match="d.jsonl line 1: tokens must be a list of strings"):
        data.load_rows(p, "token-cls")


def test_load_rows_wraps_bad_utf8(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_bytes(json.dumps({"id": 0, "text": "ok", "label": "a"}).encode() + b"\n" + b'{"id": 1, "text": "\xff"}\n')
    with pytest.raises(DataError, match="d.jsonl line 2: not valid UTF-8"):
        data.load_rows(p, "seq-cls")

