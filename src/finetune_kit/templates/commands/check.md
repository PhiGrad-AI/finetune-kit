# finetune-kit: check

Check the experiment's data, explain every finding in plain words, create and freeze the
train / validation / test splits, and write `data-report.md`.

## Before you start

`experiments/<exp>/experiment.md` must exist. If it does not, the setup stage has not run:
tell the user to run `{{CMD}}setup` first, and stop.

## How to work in this stage

- First read `.finetune-kit/memory/research-rules.md` and follow it throughout.
- Ask **one question at a time** and wait for the answer. Use plain language and explain any
  ML term in a few words the first time it comes up.
- Act only through `finetune-kit …` commands. Each prints one JSON object: read it before you
  go on, and base what you tell the user on it, not on guesses.
- Never edit files that finetune-kit writes (`experiment.yaml`, `search-space.yaml`,
  `state.json`, `results.tsv`, `data-report.json`, `splits/`, `runs/`, `final/`,
  `.finetune-kit/memory/`). Never open anything under `~/.finetune-kit/` or
  `experiments/*/splits/`, and do not open the user's dataset files.
- If a command prints `{"error": "..."}`, tell the user in plain words what went wrong and what
  would fix it. Fix it yourself when the fix is yours to make (for example a mistyped option);
  otherwise stop and wait for the user.
- Training runs detach and return at once. Poll a run with
  `finetune-kit wait <run_id> --exp <exp> --timeout 540`. If it returns `"status": "queued"` or
  `"running"`, call it again, in a loop, until the status is `done`, `crash` or `timeout`.
  Give the user a short progress line now and then, not after every poll.

**Which experiment.** `<exp>` below is the experiment folder's number (e.g. `001`). If the user
named one, use it. Otherwise use the highest-numbered folder in `experiments/` and say which one
you picked.

## Steps

1. Run `finetune-kit check-data <exp>`. It validates the data, splits it (if the user gave one
   file), moves the test split into a private vault outside the project, freezes the splits and
   writes `data-report.json`. It may take a minute the first time (it downloads the model's
   tokenizer).

   If the error says the splits are already frozen, the check ran before: read the existing
   `experiments/<exp>/data-report.json` (reading is fine) and continue with step 2. Only offer
   `--resplit` if the user wants to change the data, and warn that it resets the experiment's
   progress (bake-off, noise floor, best run).

2. **Explain the report**, briefly and in plain words, one topic per line:
   - `n_rows`: rows in train, validation (`val`) and test.
   - `split_origin`: where each of train, validation and test came from (a published split, a
     file, or carved from train). If any part was carved from train, say so and give the reason
     from the matching `warnings` line ("<role> carved from train: <reason>").
   - `ignored_splits`, if not empty: published splits of the dataset that were not used. Tell
     the user which ones and that they play no part.
   - `label_dist` and `minority_share`: how the labels are spread. If the smallest class is
     rare (say under 30%), mention that the hparams stage will suggest settings for that.
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
   has no `label`"), and stop. Tell them to rerun `{{CMD}}check` after fixing it; if the splits
   were already frozen, the rerun uses `finetune-kit check-data <exp> --resplit`.

4. **Small validation set.** If a warning says the validation set is too small for the metric,
   explain why it matters: the metric then rests on a handful of examples, so one example
   flipping changes the score, and the research would chase noise instead of real gains.
   - If finetune-kit split a single file, offer to re-split with a bigger validation share:
     `finetune-kit check-data <exp> --resplit --val-frac X`. Pick X so validation gets about the
     number of positives the warning asks for: X ≈ needed ÷ (all positives across train, val
     and test), rounded up to the next 0.05. Test stays at 10%. Above 0.3 the training set gets
     noticeably smaller: say so and let the user decide. If they agree, run it and explain the
     new report (steps 2–3).
   - If the user gave separate validation and test files, `--val-frac` does not apply: they need
     a bigger validation file and a new experiment via `{{CMD}}setup`. Let them decide whether to
     go on anyway.

5. **Write `data-report.md`** in the experiment folder with these sections: *Summary* (two or
   three sentences), *Splits* (row counts, how they were made, that test sits in the vault),
   *Labels*, *Lengths and truncation*, *Duplicates and leakage*, *Warnings* (each in plain
   words, with what was decided), and *Next*.

## Finish

Read `experiment.md`. If the user chose a bake-off, the next command is `{{CMD}}bakeoff`;
otherwise it is `{{CMD}}hparams`. Tell the user which, in one line.
