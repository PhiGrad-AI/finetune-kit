# finetune-kit: promote

Retrain the chosen recipe on the full training data, score the test set **exactly once**, and
export the final model with its threshold, speed figures and a standalone `predict.py`.

## Before you start

{{ORDER_CHECK}}

If `report.md` is missing, say so; `finetune-kit compare` has all you need to choose the run.

## How to work in this stage

- First read `.finetune-kit/memory/research-rules.md` and follow it throughout.
- Ask **one question at a time** and wait for the answer. Use plain language and explain any
  ML term in a few words the first time it comes up.
- Act only through `finetune-kit …` commands. Each prints one JSON object: read it before you
  go on, and base what you tell the user on it, not on guesses.
- Never edit files that finetune-kit writes (`experiment.yaml`, `search-space.yaml`,
  `state.json`, `results.tsv`, `data-report.json`, `splits/`, `runs/`, `final/`,
  `.finetune-kit/memory/`). Never open anything under `~/.finetune-kit/`, `~/.cache/huggingface/`
  or `experiments/*/splits/`, and do not open the user's dataset files.
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

1. **Already done?** If `experiments/<exp>/final/test_metrics.json` exists, the experiment was
   promoted before and its test set is used up. Do not promote again: read
   `final/test_metrics.json`, `final/threshold.json` (if present) and `final/throughput.json`,
   summarize them (step 6), and finish.
   If `finetune-kit status <exp>` shows a `promote` run queued or running, wait for it (step 5).

2. **Choose the run.** Run `finetune-kit compare <exp>`. Recommend the `best` run, and show its
   description, config and validation mean ± std. Only a `keep` or `keep-simpler` run, or the
   noise run (the default recipe), can be promoted. Ask the user which run to promote. Seeds for
   the final retrain default to 3 (`--seeds N` to change); the best seed on validation is kept.

3. **Warn and ask for explicit confirmation.** Tell the user, plainly:
   - The test set is scored **exactly once**. The number it gives is final, even if it is
     disappointing; there is no second try for this experiment.
   - The run must be chosen on validation results (as above), never on test results.
   - This retrains on all the training data, which takes longer than a research run.

   Ask: "Promote <run id> and score the test set once? Please answer yes to go ahead." Go on
   only after a clear yes. Anything else: do not promote.

4. **Promote.** Run `finetune-kit promote <exp> <run id> [--seeds N]`. It returns a new
   `run_id` for the final retrain.

5. **Wait** for that run id with the wait loop. When it ends, the reply's `promote` object holds
   `test_metrics`, `threshold`, `val_metrics`, `throughput` and `final` (the folder). If it has
   an `error`:
   - before test scoring (no `test_metrics`): the test set is still unused; explain the error,
     fix the cause with the user, and promote again;
   - "export failed after test scoring": the test set is used up; explain what failed and stop.

6. **Summarize** from `final/test_metrics.json`, `final/threshold.json` and
   `final/throughput.json` (reading them is fine):
   - test score of the objective (and guardrail) next to the validation mean from research. A
     test score a bit below validation is normal; a large gap means the many research runs
     over-fitted the validation set, and is worth saying.
   - the threshold, if any: its value, which target it meets (for example recall 0.995 on
     validation), and what it means ("rows scoring at or above it are flagged positive"); the
     test recall, precision and skip rate at that threshold (`at_threshold`).
   - speed: rows per second on the training device and on CPU, about how many rows per hour
     that is (rows per second × 3600), latency per batch, and model size in MB.

7. **How to use the model.** Show:

   ```
   python experiments/<exp>/final/predict.py --input rows.jsonl --output scores.jsonl [--device cpu]
   ```

   It needs only `torch` and `transformers` (plus `peft` if the recipe uses LoRA), not
   finetune-kit. seq-cls input rows need `"text"`; output rows are `{"id", "label", "score"}`,
   plus `"flag"` when there is a threshold. token-cls input rows need `"tokens"`; output rows
   are `{"id", "tags"}`. To deploy, copy the whole `final/` folder.

## Finish

The experiment is complete. `finetune-kit next` returns `setup`; in the hand-off, say the
experiment is done and ask whether the user wants to start a new experiment.

{{FINISH}}
