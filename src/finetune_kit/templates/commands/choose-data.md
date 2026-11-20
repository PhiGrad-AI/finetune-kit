# finetune-kit: choose-data

Step 2 of 14. Ask where the training examples come from, look at them with finetune-kit (never
by opening the files), confirm which column is which, and record the answer.

## Before you start

{{ORDER_CHECK}}

If `set data` says the data is checked and locked, the data can no longer change for this
experiment: say so, and that changing it needs a new experiment (`{{CMD}}choose-task`).

## How to work in this step

{{HOW_TO_WORK}}

## Steps

1. **Where from.** Read the task from `experiments/<exp>/experiment.yaml` (reading is fine).
   Ask: "Where does your data come from?" Offer three options:
   - (a) **one file**: finetune-kit splits it 80/10/10 into train / validation / test;
   - (b) **three files**: train, validation and test (give both validation and test, or neither);
   - (c) a **Hugging Face dataset id**, for example `imdb` or `user/dataset` (optionally with a
     config name, `hf:<id>:<config>`, and a revision).

   Accepted file formats: `.jsonl`, `.csv`, `.tsv`, `.parquet`. CSV and TSV work for sequence
   classification only. Explain in one line that the test part is locked away and scored only
   once, at the very end. For a Hugging Face dataset, say that its published train / validation /
   test splits are used as they are, and that a validation or test split that is missing,
   empty or unlabelled is carved out of train instead. Private or gated datasets use the user's
   existing Hugging Face login (`hf auth login` or `huggingface-cli login`); never ask for a
   token. You may check that files exist (`ls`), but do not open them. Wait for the answer.

2. **Look at it.** Run `finetune-kit inspect-data` with the same data. The tool looks at the data
   so you do not have to; it prints only counts, column names and label values, never row text:
   - files: `finetune-kit inspect-data --train <path> [--val <path> --test <path>]`
   - Hugging Face: `finetune-kit inspect-data --source hf:<id>[:<config>] [--revision <rev>]`

   Its JSON has `splits` (row counts and role), `columns` (name and type), `label_candidates`
   (columns that look like labels, with their distinct values), `suggested_columns` and
   `tasks_possible`. Show the user, in plain words: the splits and row counts, the columns, and
   the label values. If the experiment's task is not in `tasks_possible`, say so: the data does
   not fit that task, so either other data is needed or a new experiment with the other task
   (`{{CMD}}choose-task`). An error from `inspect-data` (a missing file, an unknown dataset, a
   login needed) is explained and fixed with the user before you go on.

3. **Which column is which.** Propose the `suggested_columns` mapping (which column is the text,
   which is the label, and so on) and ask them to confirm or correct it. Wait for the answer.

4. **Record it.** Run:

   ```
   finetune-kit set data <exp> (--train <path> [--val <path> --test <path>] | --source hf:<id>[:<config>] [--revision <rev>])
       [--columns '{"text": "<col>", "label": "<col>"}']
   ```

   `--columns` takes the confirmed mapping as JSON (keys `text`, `label`, `id`, `tokens`, `tags`,
   `group_id`); leave it out when the column names already match. With `--source`, finetune-kit
   pins the dataset's exact revision now, so later steps use the same data. On an error, explain
   it, ask the one question that fixes it, and run `set data` again.

5. **Update `experiment.md`:** fill in the *Data* row and its reason.

## Finish

{{FINISH}}
