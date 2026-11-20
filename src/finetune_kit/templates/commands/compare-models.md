# finetune-kit: compare-models

Step 6 of 14, only when the user chose to compare 2-3 models. Train each candidate model with its
default settings on the same data, rank them, let the user pick one, fix it for the experiment,
and write `bakeoff.md`.

## Before you start

{{ORDER_CHECK}}

`next` names `compare-models` as the step only when a comparison was chosen in step 3
(the `bakeoff` field of `experiment.yaml`) and no model is chosen yet. If it names a later
step, the comparison is skipped or already done for this experiment. Say so and use the hand-off
in the Finish section, unless the user now asks for a comparison.

- **Late comparison.** If the user opts in now although none was chosen in step 3, ask which
  2–3 models and run it through to `finetune-kit bakeoff choose <exp> <model>` and `bakeoff.md`,
  so `finetune-kit next` agrees at Finish. If the comparison is abandoned midway, tell the user it
  is incomplete and that `finetune-kit next` will move on without it.
- **Too late for a comparison.** Once the experiments have started (step 11 measured the
  starting point), the model is locked: comparing models needs a new experiment
  (`{{CMD}}choose-task`), and `bakeoff choose` will refuse. Say so instead.
- **Chosen in step 3, user declines now.** If `next` names `compare-models` and the user
  declines, say a comparison was chosen and the experiment cannot move past it until a model is
  chosen (`finetune-kit bakeoff choose <exp> <model>` with the first model counts as choosing).

## How to work in this step

{{HOW_TO_WORK}}

## Steps

1. **Resume check.** Run `finetune-kit bakeoff <exp>` (no `--models`). If `bakeoff` already
   holds a result and `pending` is 0, go to step 4. If `pending` is above 0, find the runs with
   `finetune-kit status <exp>` and wait for them (step 3).

2. **Start.** Tell the user the candidates and that each one is trained `seeds` times (from
   `experiment.yaml`) with its default settings; this can take hours. Then run
   `finetune-kit bakeoff <exp> --models <a>,<b>[,<c>]`. It returns the `run_ids`.

3. **Wait** for every run id in turn with the `finetune-kit wait` loop. When a run ends, give the
   user one line: model, status, and the mean of its seeds' `objective`. If a run crashed, show
   the `error` of its seeds in plain words.

4. **Result.** Run `finetune-kit bakeoff <exp>`. The `bakeoff` object has `ranking` (per model:
   `mean`, `std`, `train_s`), `winner`, `clear`, `reason`, `runs` and `failed`. Show a small
   table: model, mean ± std of the objective, training time per seed in minutes. Remember the
   metric's direction: for `log_loss`, lower is better.
   - `clear: true`: the winner beats the runner-up by more than the noise bar, so the difference
     is real.
   - `clear: false`: the top models are tied within noise; finetune-kit then recommends the
     faster one, because research will run many times on it.
   - Name any `failed` models and why they failed.

5. **Confirm.** Recommend the `winner` and ask the user to confirm it or pick another model. Wait.

6. **Fix the model.** Run `finetune-kit bakeoff choose <exp> <model>`. Explain that the model is
   now fixed for this experiment.

7. **Write `bakeoff.md`** in the experiment folder: candidates, the ranking table, whether the
   result was clear or a tie and why, the chosen model, and *Next*.

## Finish

{{FINISH}}
