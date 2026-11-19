# finetune-kit: CSV, Parquet and Hugging Face data sources

Short design, follow-up to `2026-11-14-finetune-kit-design.md`.

## Why

Most public datasets come already split into train / validation / test, as CSV or Parquet
or as a Hugging Face dataset, with their own column names and no `id` field. Today
finetune-kit reads only JSONL with `id`, so every such dataset needs a hand conversion
before setup. finetune-kit already uses a dataset's own splits when given three files;
this change makes that work straight from the formats people actually have.

## Goals

- Accept local `.csv`, `.tsv`, `.parquet` and `.jsonl` files for `--train` / `--val` / `--test`.
- Accept a Hugging Face dataset id (`hf:S-Labs/prompt-injection-dataset`, optionally
  `hf:glue:sst2` for a config) as the data source, using its own splits.
- Map the user's column names to finetune-kit's fields; create ids when there are none.
- Keep every existing guarantee: the test split is locked in the vault and scored once,
  splits are frozen and checksummed, the agent never reads row contents.

## Non-goals

- Streaming or datasets larger than memory.
- Joining several datasets, or filtering rows.
- Uploading anything. Data only ever comes down from the Hub.

## Approaches

- **A. Local CSV/Parquet only.** The user downloads files themselves. Smallest change.
- **B. A plus Hugging Face ids.** finetune-kit downloads the dataset and finds its splits.
- **Chosen: both (B includes A)**, at the user's request: "both should work".

## Design

**One loader, many sources.** `data.load_rows(path, task, columns)` dispatches on the file
suffix (`.jsonl`, `.csv`, `.tsv`, `.parquet`). It renames the user's columns to the
standard fields (`text`/`label`, or `tokens`/`tags`, plus optional `id`/`group_id`) and
then applies today's validation unchanged. Missing ids become `<split>-<row number>`.

**Column mapping in experiment.yaml.** `data.columns: {text: prompt, label: is_injection}`.
When the columns are already called `text`/`label` (or `tokens`/`tags`) no mapping is
needed. Integer labels with names (a Hugging Face `ClassLabel`) are stored as their names,
so results say `injection`, not `1`.

**Hugging Face source.** `data.source: hf:<id>[:<config>]` plus `data.revision` (the commit
sha, pinned at setup so the experiment is reproducible). Setup downloads with the
`datasets` library (new dependency) into `~/.finetune-kit/hf-cache/`, using the user's
existing Hugging Face login for private or gated datasets; finetune-kit never asks for or
stores a token. Split names are matched by alias: `train`; `validation`/`valid`/`val`/`dev`;
`test`.

**Which splits are used.**
- train + validation + test, all labelled: used as they are.
- train + test (no validation): validation is carved from train (10%, stratified).
- train + validation (no test, or a test split without labels, such as GLUE's `-1`):
  test is carved from train, and the report says the published test split was not used.
- train only: today's 80/10/10 split.
The data report states which of these happened.

**Inspecting without reading.** New `finetune-kit inspect-data <source>` prints JSON: splits
and row counts, column names and types, the label values with counts, and a suggested
column mapping. It never prints row text, so the setup stage can show the user what is
there and ask which column is the text and which the label without the agent opening the
data.

**Setup and check stages.** The setup prompt asks for a file, three files, or a Hugging Face
id; runs `inspect-data`; proposes the column mapping and asks the user to confirm it.
`check-data` then loads, converts, validates and freezes splits exactly as today (the
existing cross-split duplicate warning covers leakage in published splits).

## Left to the builder

- Exact CLI flag names for the column mapping and source.
- How `inspect-data` caps label listings for high-cardinality columns.
- Test fixtures: small local CSV/Parquet files, and a fake `datasets` dataset object for the
  Hugging Face path (no network in tests).
