# finetune-kit: check-data

Step 5 of 14. Check the experiment's data, explain every finding in plain words, create and
freeze the train / validation / test splits, and write `data-report.md`.

## Before you start

{{ORDER_CHECK}}

## How to work in this step

{{HOW_TO_WORK}}

## Steps

1. Explain in one line what happens: finetune-kit checks the data for problems, splits it, and
   locks the test part away; after this the data, model and goal answers can no longer change.
   Then run `finetune-kit check-data <exp>`. It validates the data, splits it (if the user gave one
   file), moves the test split into a private vault outside the project, freezes the splits and
   writes `data-report.json`. It may take a minute the first time (it downloads the model's
   tokenizer).

   If the error says the splits are already frozen, the check ran before: read the existing
   `experiments/<exp>/data-report.json` (reading is fine) and continue with step 2 below. Only offer
   `--resplit` if the user wants to change the data, and warn that it resets the experiment's
   progress (model comparison, starting-point measurement, best run).

2. **Explain the report**, briefly and in plain words, one topic per line:
   - `n_rows`: rows in train, validation (`val`) and test.
   - `split_origin`: where each of train, validation and test came from (a published split, a
     file, or carved from train). If any part was carved from train, say so and give the reason
     from the matching `warnings` line ("<role> carved from train: <reason>").
   - `ignored_splits`, if not empty: published splits of the dataset that were not used. Tell
     the user which ones and that they play no part.
   - `label_dist` and `minority_share`: how the labels are spread. If the smallest class is
     rare (say under 30%), mention that Step 7 of 14 — Choose the loss and Step 8 of 14 — Choose
     the settings will suggest options for that.
   - `token_len` (`p50`, `p95`, `max`) and `n_over_max_len`: text length in model tokens. Rows
     longer than the model's max length are cut off (truncated) during training; say how many,
     and whether it matters (a few rows: fine; many rows: the end of those texts is never seen,
     so the user may want to split long texts into windows).
   - `n_exact_dupes`: identical rows; usually harmless but worth cleaning.
   - `n_cross_split_dupes` and `n_group_leak`: the same text or group in more than one split.
     This is leakage: validation scores look better than they really are. Recommend fixing it.
   - `unseen_labels`: labels in validation or test that never appear in training.
   - `entity_token_share` (token-cls): share of tokens that belong to an entity.
   - each line in `warnings`.

3. **Errors.** If `errors` is not empty (or the command failed with a data error), explain each
   one, propose a concrete fix the user makes in their own data files (for example "line 512
   has no `label`"), and stop. Tell them to rerun `{{CMD}}check-data` after fixing it; if the splits
   were already frozen, the rerun uses `finetune-kit check-data <exp> --resplit`.

4. **Small validation set.** If a warning says the validation set is too small for the metric,
   explain why it matters: the metric then rests on a handful of examples, so one example
   flipping changes the score, and the research would chase noise instead of real gains.
   - If finetune-kit split a single file, offer to re-split with a bigger validation share:
     `finetune-kit check-data <exp> --resplit --val-frac X`. Pick X so validation gets about the
     number of positives the warning asks for: X ≈ needed ÷ (all positives across train, val
     and test), rounded up to the next 0.05. Test stays at 10%. Above 0.3 the training set gets
     noticeably smaller: say so and let the user decide. If they agree, run it and explain the
     new report (steps 2–3 above).
   - If the user gave separate validation and test files, `--val-frac` does not apply: they need
     a bigger validation file and a new experiment via `{{CMD}}choose-task`. Let them decide
     whether to go on anyway.

5. **Write `data-report.md`** in the experiment folder with these sections: *Summary* (two or
   three sentences), *Splits* (row counts, how they were made, that test sits in the vault),
   *Labels*, *Lengths and truncation*, *Duplicates and leakage*, *Warnings* (each in plain
   words, with what was decided), and *Next*.

## Finish

{{FINISH}}
