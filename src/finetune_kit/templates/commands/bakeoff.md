# finetune-kit: bakeoff

Train each candidate model with its default settings on the same data, rank them, let the user
pick one, fix it for the experiment, and write `bakeoff.md`.

## Before you start

{{ORDER_CHECK}}

`next` names `bakeoff` as the stage only when a bake-off was requested at setup
(the `bakeoff` field of `experiment.yaml`) and no model is chosen yet. If it names a later
stage, the bake-off is skipped or already done for this experiment. Say so and use the hand-off
in the Finish section, unless the user now asks for a bake-off.

- **Late bake-off.** If the user opts in now although none was requested at setup, ask which
  2–3 models and run it through to `finetune-kit bakeoff choose <exp> <model>` and `bakeoff.md`,
  so `finetune-kit next` agrees at Finish. If the bake-off is abandoned midway, tell the user it
  is incomplete and that `finetune-kit next` will move on to `hparams` without it.
- **Too late for a bake-off.** Once the baseline stage has run, the model is locked: a bake-off
  needs a new experiment (`{{CMD}}setup`), and `bakeoff choose` will refuse. Say so instead.
- **Requested at setup, user declines.** If `next` names `bakeoff` and the user declines, say the
  bake-off was requested at setup and the experiment cannot move past it until a model is chosen
  (`finetune-kit bakeoff choose <exp> <model>` with the setup model counts as choosing).

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

5. **Confirm.** Recommend the `winner` and ask the user to confirm or pick another model.

6. **Fix the model.** Run `finetune-kit bakeoff choose <exp> <model>`. Explain that the model is
   now fixed for this experiment.

7. **Write `bakeoff.md`** in the experiment folder: candidates, the ranking table, whether the
   result was clear or a tie and why, the chosen model, and *Next*.

## Finish

{{FINISH}}
