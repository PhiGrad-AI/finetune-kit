# finetune-kit data sources Implementation Plan

**Goal:** Let finetune-kit read local CSV / TSV / Parquet files and Hugging Face datasets (with their own splits) in addition to JSONL, with a column mapping and generated ids.

**Architecture:** A new module `finetune_kit/sources.py` turns any source into the standard row dicts (`id`, `text`, `label` or `id`, `tokens`, `tags`, optional `group_id`); `data.py` keeps validation. `experiment.DataPaths` gains `columns`, `source` and `revision`. `commands.setup_experiment` and `commands.check_data` go through one `sources.load_source(...)` call. A new `inspect-data` command describes a source without printing row text. Frozen splits, checksums and the test vault are unchanged.

**Tech Stack:** existing stack plus `datasets` (Hugging Face) and `pyarrow` (Parquet).

**Spec:** `docs/design/specs/2026-11-14-data-sources-design.md` (parent: `docs/design/specs/2026-11-14-finetune-kit-design.md`)

## Global Constraints

- Every CLI command prints exactly one JSON object; errors print `{"error": "<message>"}` and exit 2. Library errors from `datasets`/`huggingface_hub`/`pyarrow`/`csv` are re-raised as `ValueError` with a one-line message (never a traceback).
- Hugging Face source string: `hf:<repo_id>` or `hf:<repo_id>:<config>` (e.g. `hf:S-Labs/prompt-injection-dataset`, `hf:glue:sst2`). `repo_id` may contain one `/`.
- Hugging Face cache dir: `kit_home() / "hf-cache"`. Never ask for, read or store a token; the user's existing `huggingface_hub` login is used implicitly.
- Revision: resolved to a commit sha at `new` time and stored in `experiment.yaml` as `data.revision`; `check-data` loads that exact revision.
- Split aliases: train = `train`; val = `validation`, `valid`, `val`, `dev`; test = `test`. Case-insensitive. Any other split is ignored and listed in the report.
- Generated ids: `<role>-<row number, 1-based, zero-padded to 6>` e.g. `train-000001`, `val-000001`, `test-000001`. A user id column (mapped as `id`) is used as-is; ids are stringified.
- A label is "missing" when it is `None`, empty string, or the integer `-1`.
- ClassLabel integers are replaced by their names (seq-cls `label`, token-cls `tags`).
- CSV/TSV support seq-cls only (token-cls needs list columns: JSONL or Parquet or HF).
- Local three-file rule is unchanged: give both `--val` and `--test`, or neither.
- Tests never touch the network: the HF path is tested by monkeypatching `sources._load_dataset` and `sources._resolve_revision`.
- `inspect-data` never prints row text or token values.

## Review Focus

1. **CSV quirks** (UTF-8 BOM, quoted commas/newlines, empty text cell, numeric labels read as strings `"0"`/`"1"`) → loads correctly or fails with `train.csv row N: …`; labels from CSV and JSONL of the same dataset compare equal (both stringified by `label_map`). Test in Task 1.
2. **Column mapping typo** (`--columns '{"text":"promt"}'`) → error naming the missing column and listing the available ones, raised at `new`, before anything is written. Test in Task 1.
3. **HF test split without labels** (all `-1`, GLUE style) → test carved from train, `split_origin.test == "carved from train (published test has no labels)"`. Test in Task 2.
4. **HF dataset not found / offline / gated** → JSON error, no traceback, no experiment dir created. Test in Task 3.
5. **Generated ids colliding with real ids across splits** → ids are unique per split file (existing rule); generated ids carry the role prefix so train/val/test never collide. Test in Task 2.

---

### Task 1: Local CSV / TSV / Parquet and column mapping

**Files:**
- Create: `src/finetune_kit/sources.py`
- Modify: `src/finetune_kit/data.py` (split `load_rows` into reading + `validate_rows`)
- Modify: `src/finetune_kit/experiment.py` (`DataPaths.columns`)
- Modify: `src/finetune_kit/commands.py` (`setup_experiment`, `check_data` pass columns)
- Modify: `src/finetune_kit/cli.py` (`new --columns`), `pyproject.toml` (add `pyarrow`)
- Test: `tests/test_sources.py`, `tests/test_cli_commands.py`

**Interfaces:**
- Produces: `sources.read_table(path: Path) -> list[dict]` (by suffix `.jsonl`, `.csv`, `.tsv`, `.parquet`; other suffix → `ValueError("unsupported data file type .xyz; use .jsonl, .csv, .tsv or .parquet")`).
- Produces: `sources.normalize(raw: list[dict], task: str, columns: dict[str, str], role: str, where: str, label_names: dict[str, list[str]] | None = None) -> list[dict]` — renames mapped columns (keys are standard field names, values are source column names), drops unmapped extra columns, adds generated ids, applies ClassLabel names. Missing mapped column → `ValueError(f"{where}: no column {col!r}; columns are {sorted(available)}")`.
- Produces: `data.validate_rows(rows: list[dict], task: str, where: str) -> list[dict]` — today's per-row checks, messages `"{where} row {n}: …"` (JSONL keeps `line {n}`).
- Produces: `data.load_rows(path: Path, task: str, columns: dict | None = None, role: str = "train") -> list[dict]` (unchanged call sites keep working).
- Produces: `DataPaths.columns: dict[str, str] = {}`.
- Default mapping when `columns` is empty: identity for whichever of `id`, `text`, `label`, `tokens`, `tags`, `group_id` exist.

- [ ] **Step 1: Write failing tests** in `tests/test_sources.py`:
  - `test_csv_with_bom_quotes_and_newlines` — write `﻿text,label\n"a, b",0\n"line1\nline2",1\n` → two rows, texts `"a, b"` and `"line1\nline2"`, ids `train-000001`, `train-000002`.
  - `test_tsv_reads` — tab-separated equivalent loads.
  - `test_parquet_reads` — write with `pyarrow.parquet.write_table`; token-cls with list columns `tokens`/`tags` loads.
  - `test_column_mapping` — CSV `prompt,is_injection` with `columns={"text":"prompt","label":"is_injection"}` → rows have `text`, `label`, `id`, and no `prompt`.
  - `test_missing_mapped_column_lists_available` — `columns={"text":"promt"}` → `ValueError` containing `"no column 'promt'"` and `"['is_injection', 'prompt']"`.
  - `test_empty_text_cell_errors_with_row` — `ValueError` matching `"train.csv row 2: text must be a non-empty string"`.
  - `test_csv_token_cls_refused` — `ValueError` containing `"CSV/TSV support seq-cls only"`.
  - `test_unsupported_suffix` — `.xlsx` → the exact message above.
  - `test_existing_jsonl_unchanged` — a JSONL with ids loads exactly as before.
- [ ] **Step 2:** Run `uv run pytest tests/test_sources.py -q` — expect failures (module missing).
- [ ] **Step 3:** Implement. CSV via `csv.DictReader` with `encoding="utf-8-sig"`, `newline=""`; Parquet via `pyarrow.parquet.read_table(path).to_pylist()`. `load_rows` = `validate_rows(normalize(read_table(path), ...), task, path.name)`.
- [ ] **Step 4:** Add `columns` through `setup_experiment(..., columns: dict | None = None)` → `DataPaths(columns=...)`; `check_data` passes `cfg.data.columns` and the role to each `load_rows`. CLI `new --columns` takes a JSON object (parsed with `cmd.parse_params(columns, "--columns")`). Test in `tests/test_cli_commands.py`: `test_new_with_csv_and_columns` creates an experiment from CSV with a mapping and `check-data` freezes JSONL splits containing `text`/`label`/`id`; `test_new_bad_column_writes_nothing` asserts no `experiments/001-*` dir exists after the error.
- [ ] **Step 5:** `uv run pytest -q tests/test_sources.py tests/test_data.py tests/test_cli_commands.py` — PASS.
- [ ] **Step 6:** Commit `feat: read CSV, TSV and Parquet data with a column mapping`.

### Task 2: Hugging Face loading and split resolution

**Files:**
- Modify: `src/finetune_kit/sources.py`, `pyproject.toml` (add `datasets`)
- Test: `tests/test_sources.py`

**Interfaces:**
- Consumes: `sources.normalize`, `data.validate_rows`, `data.make_splits` (Task 1 / existing).
- Produces: `sources.parse_hf(spec: str) -> tuple[str, str | None]` (`"hf:a/b:cfg"` → `("a/b", "cfg")`; malformed → `ValueError("Hugging Face source must be hf:<id> or hf:<id>:<config>")`).
- Produces: `sources._resolve_revision(repo_id: str, revision: str | None) -> str` (module-level hook; uses `huggingface_hub.HfApi().dataset_info(repo_id, revision=revision).sha`).
- Produces: `sources._load_dataset(repo_id: str, config: str | None, revision: str) -> dict[str, tuple[list[dict], dict]]` (module-level hook; split name → (rows, features dict); uses `datasets.load_dataset(..., cache_dir=str(kit_home()/"hf-cache"))`; features from `ds.features`).
- Produces: `sources.pick_splits(names: list[str]) -> dict[str, str]` (role → split name, per the alias list; `"train"` missing → `ValueError(f"no train split; splits are {names}")`).
- Produces: `sources.label_names(features: dict, columns: dict[str, str], task: str) -> dict[str, list[str]]` (ClassLabel names for the mapped label/tags column, else `{}`).
- Produces: `sources.load_hf(spec: str, revision: str, task: str, columns: dict) -> Loaded` where `Loaded` is a dataclass `(train: list[dict], val: list[dict] | None, test: list[dict] | None, origin: dict[str, str], ignored_splits: list[str], test_unlabelled: bool)`.
- Produces: `sources.resolve_splits(loaded: Loaded, task: str, val_frac: float) -> tuple[list, list, list, dict]` returning train, val, test and `split_origin`:
  - all three present and test labelled → as-is; origin `{"train": "published", "val": "published", "test": "published"}`.
  - val missing → val carved from train with `make_splits(train, task, frac=(1 - val_frac, val_frac, 0))`; origin val `"carved from train"`.
  - test missing → test carved from train with `frac=(0.9, 0, 0.1)`; origin test `"carved from train (no published test split)"`.
  - test present but every label missing → same carve; origin test `"carved from train (published test has no labels)"`.
  - only train → existing `make_splits(train, task, frac=(1 - 0.1 - val_frac, val_frac, 0.1))`; all non-train origins `"carved from train"`.
  Carving happens before `validate_rows` on val/test so unlabelled rows are never validated.

- [ ] **Step 1: Write failing tests** (fake `_load_dataset` / `_resolve_revision` via `monkeypatch`, features as `{"label": {"names": ["benign", "injection"]}}`-shaped stand-ins built with real `datasets.ClassLabel` objects):
  - `test_parse_hf` — three valid forms and one invalid.
  - `test_pick_splits_aliases` — `["train", "dev", "test"]` → val=`dev`; `["TRAIN", "Validation"]` → train/val; `["foo"]` → error.
  - `test_classlabel_names_applied` — label 1 → `"injection"`.
  - `test_all_published_used_as_is` — sizes 8/2/2 in → 8/2/2 out, origin all `"published"`.
  - `test_unlabelled_test_carved` — test labels all `-1` → test carved from train; origin message as in Review Focus 3.
  - `test_missing_val_carved` — train+test only → val carved, test published.
  - `test_generated_ids_unique_across_splits` — union of ids across the three splits has no duplicates.
  - `test_ignored_splits_reported` — extra split `"unsupervised"` → `ignored_splits == ["unsupervised"]`.
- [ ] **Step 2:** Run — expect failures.
- [ ] **Step 3:** Implement. Wrap `datasets`/`huggingface_hub` exceptions in `_load_dataset`/`_resolve_revision` as `ValueError(f"could not load {spec}: {e}")` (first line of the message only).
- [ ] **Step 4:** `uv run pytest tests/test_sources.py -q` — PASS.
- [ ] **Step 5:** Commit `feat: load Hugging Face datasets and resolve their splits`.

### Task 3: Wire Hugging Face sources into `new` and `check-data`

**Files:**
- Modify: `src/finetune_kit/experiment.py`, `src/finetune_kit/commands.py`, `src/finetune_kit/cli.py`, `src/finetune_kit/data.py` (report field)
- Test: `tests/test_cli_commands.py`

**Interfaces:**
- Consumes: Task 1 and Task 2 functions.
- Produces: `DataPaths.train: str | None = None`, `DataPaths.source: str | None = None`, `DataPaths.revision: str | None = None`, with a model validator: exactly one of `train` / `source` (`ValueError("give --train or --source, not both")` / `"give --train or --source"`); `val`/`test` only with `train`.
- Produces: `sources.load_source(cfg_data: DataPaths, root: Path, task: str, val_frac: float) -> tuple[list, list, list, dict]` — the single entry point used by `check_data` (local: existing three-file rule + `make_splits`, origin `"provided"` / `"carved from train"`; HF: `load_hf` + `resolve_splits`). Also `sources.load_train(cfg_data, root, task) -> list[dict]` used by `setup_experiment` for label checks.
- Produces: `data-report.json` gains `"split_origin": {...}` and `"ignored_splits": [...]`; when any split was carved, a warning `"<role> carved from train: <reason>"` is appended.
- CLI: `new --source hf:… [--revision REV]`; `--train` becomes optional. Revision is resolved and stored at `new`.

- [ ] **Step 1: Write failing tests** (fakes as in Task 2):
  - `test_new_from_hf_pins_revision` — `experiment.yaml` has `data.source` and `data.revision == "abc123"` (fake sha).
  - `test_check_data_from_hf_freezes_published_splits` — `splits/train.jsonl` row count equals fake train size; vault test equals fake test size; report `split_origin` all `"published"`.
  - `test_hf_not_found_is_json_error` — fake raises → CLI exit 2, stdout `{"error": ...}` containing `"could not load"`, no experiment dir.
  - `test_train_and_source_both_refused` — error `"give --train or --source, not both"`.
  - `test_existing_jsonl_experiment_yaml_still_loads` — an `experiment.yaml` written by the previous version (only `data.train/val/test`) loads.
- [ ] **Step 2:** Run — expect failures.
- [ ] **Step 3:** Implement; `check_data` replaces its inline loading block with `sources.load_source(...)` and stores `split_origin`/`ignored_splits` in the report.
- [ ] **Step 4:** `uv run pytest -q tests/test_cli_commands.py tests/test_experiment.py tests/test_sources.py` — PASS.
- [ ] **Step 5:** Commit `feat: create experiments from Hugging Face datasets`.

### Task 4: `inspect-data` command

**Files:**
- Modify: `src/finetune_kit/sources.py`, `src/finetune_kit/cli.py`
- Test: `tests/test_sources.py`, `tests/test_cli.py`

**Interfaces:**
- Produces: `sources.inspect(source: str | None, train: str | None, val: str | None, test: str | None, root: Path, revision: str | None = None) -> dict` returning:
  `{"source": str, "revision": str | None, "splits": {name: {"n_rows": int, "role": "train"|"val"|"test"|None}}, "columns": {name: type_str}, "label_candidates": {col: {"n_distinct": int, "values": {value: count}}}, "suggested_columns": {...}, "tasks_possible": [...]}`.
  - `type_str`: `"string"`, `"int"`, `"float"`, `"bool"`, `"list[string]"`, `"list[int]"`, `"classlabel"`, `"list[classlabel]"`, else `"other"`.
  - `label_candidates`: columns of type int / bool / classlabel / string with ≤ 50 distinct values in train; `values` lists at most 20 (most common), counts from train; ClassLabel values shown by name.
  - `suggested_columns`: text = the string column with the longest mean length; label = the label candidate named `label`/`labels`/`class`/`target`, else the one with the fewest distinct values; tokens/tags = `list[string]` + `list[classlabel|string]` columns when present.
  - Never includes row text or tokens.
- CLI: `finetune-kit inspect-data [--source hf:…] [--revision R] [--train P] [--val P] [--test P]`.

- [ ] **Step 1: Write failing tests:** `test_inspect_csv_suggests_mapping` (`prompt,is_injection` → suggested `{"text": "prompt", "label": "is_injection"}`), `test_inspect_hf_shows_classlabel_names` (values keyed `"benign"`/`"injection"`), `test_inspect_never_prints_text` (a unique sentinel string in the data does not appear in `json.dumps(result)`), `test_cli_inspect_data_json` (one JSON object, exit 0).
- [ ] **Step 2:** Run — expect failures.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** `uv run pytest -q tests/test_sources.py tests/test_cli.py` — PASS.
- [ ] **Step 5:** Commit `feat: inspect-data describes a dataset without showing its text`.

### Task 5: Stage prompt and README

**Files:**
- Modify: `src/finetune_kit/templates/commands/setup.md`, `src/finetune_kit/templates/commands/check.md`, `README.md`
- Test: `tests/test_installer.py`

- [ ] **Step 1:** In `setup.md` step 2 (Data): offer (a) one file, (b) three files, or (c) a Hugging Face dataset id; accepted formats `.jsonl`, `.csv`, `.tsv`, `.parquet` (CSV/TSV seq-cls only). Then run `finetune-kit inspect-data …`, show splits and row counts, the columns and label values, propose the suggested mapping and ask the user to confirm or correct it; pass it as `--columns` (omit when identity). For HF say the published splits are used and that a missing or unlabelled split is carved from train. Keep "do not open the data files".
- [ ] **Step 2:** In `check.md`: when reporting the data check, show `split_origin` and any `ignored_splits`.
- [ ] **Step 3:** README: data formats section (formats, `--columns`, `--source hf:…`, split rules, cache location, private datasets use your existing `huggingface-cli login`).
- [ ] **Step 4:** Add assertions in `tests/test_installer.py` that the installed setup prompt mentions `inspect-data` and `--source hf:` for each agent. Run `uv run pytest -q tests/test_installer.py` — PASS.
- [ ] **Step 5:** Commit `docs: setup offers CSV, Parquet and Hugging Face data`.

### Task 6: End-to-end

**Files:**
- Test: `tests/test_e2e.py`

- [ ] **Step 1:** Add `test_e2e_csv_source`: tiny-bert seq-cls from a CSV with a column mapping → `new` → `check-data` → one noise run (2 seeds, existing tiny settings) finishes `ok`.
- [ ] **Step 2:** Add `test_e2e_hf_source`: same flow with `--source hf:fake/ds` and the monkeypatched loader (published train/validation/test with ClassLabel names) → frozen splits use the label names, noise run finishes `ok`.
- [ ] **Step 3:** `uv run pytest -q` — full suite PASS.
- [ ] **Step 4:** Commit `test: end-to-end runs from CSV and Hugging Face sources`.
