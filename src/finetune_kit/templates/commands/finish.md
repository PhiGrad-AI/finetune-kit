# finetune-kit: finish

Step 14 of 14. Retrain the chosen recipe on the full training data, score the test set
**exactly once**, and export the final model with its threshold, speed figures and a standalone
`predict.py`.

## Before you start

{{ORDER_CHECK}}

If `report.md` is missing, say so; `finetune-kit compare` has all you need to choose the run.

## How to work in this step

{{HOW_TO_WORK}}

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

The experiment is complete. `finetune-kit next` returns `choose-task` ("this experiment is
complete; start a new one"); in the hand-off, say the experiment is done and ask whether the user
wants to start a new experiment.

{{FINISH}}
