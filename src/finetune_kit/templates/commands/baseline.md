# finetune-kit: baseline

Find a good default learning rate (when allowed), then measure the noise floor: how much the
default recipe's score varies from seed to seed. Every later decision is measured against it.
Write `baseline.md`.

## Before you start

{{ORDER_CHECK}}

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

1. **Resume check.** Run `finetune-kit compare <exp>` and `finetune-kit status <exp>`.
   - If `noise.sigma_obj` is already set, the baseline is done: if `baseline.md` is missing,
     write it from what `compare` shows (step 5), then finish.
   - If a `lr-scan` or `noise` run is queued or running, wait for it and continue from the step
     it belongs to.
   - If `runs` already has `lr-scan` rows for this experiment, skip step 2.

2. **LR scan** (only if `lr` is in the `tunable` list of `experiments/<exp>/search-space.yaml`,
   or the model's notes say "generic: lr-scan required"; otherwise say it is skipped and why).
   - Explain: five short runs (20% of the training schedule, one seed each) at learning rates
     spread over the model's safe range, to find a good starting learning rate.
   - Run `finetune-kit lr-scan <exp>`. It returns `run_ids` and `lrs`.
   - Wait for each run id in turn. Then show a table: learning rate → objective (from the
     seed's `objective`), with crashes marked. The best one (highest, or lowest for
     lower-is-better metrics) becomes the experiment's default learning rate automatically.
   - Say plainly that these are single short runs, so the scan only picks a sensible starting
     point; research refines it.
   - If every scan run crashed, explain the errors and stop.

3. **Noise floor.** Explain: the default recipe is now trained once per seed (the experiment's
   `seeds`). The spread of the scores, σ, is the noise: any later change must beat the best
   result by clearly more than that to count. Run `finetune-kit run <exp> --stage noise`, then
   wait for it with the wait loop.
   - If it ends in `crash` or `timeout`, explain the seeds' `error` and stop; once the cause is
     fixed, the noise run may be started again.

4. **Read the result.** Run `finetune-kit compare <exp>`. Report in plain words:
   - the baseline score: mean ± σ of the objective (the `noise` row in `runs`), and the
     guardrail mean and `sigma_guard` if there is a guardrail;
   - the **keep bar** (`noise.keep_bar`): "a change has to improve <metric> by more than
     <keep_bar> to be kept";
   - time per run (`minutes` of the noise row) and what that means for the research budget
     (`budget_left`): roughly how many runs fit in the hours left. Runs that take more than 3×
     the noise run's training time per seed are stopped as timeouts.
   - If σ is large compared with the gains the user hopes for, say so: small improvements will
     be hard to prove, and more seeds per run would help (that needs a new experiment).

5. **Write `baseline.md`** in the experiment folder from `.finetune-kit/templates/baseline.md`.

## Finish

Tell the user research can now run on its own for hours, and can be stopped and resumed at any
time.

{{FINISH}}
