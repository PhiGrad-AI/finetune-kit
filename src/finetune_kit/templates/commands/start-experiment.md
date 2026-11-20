# finetune-kit: start-experiment

Step 11 of 14. First measure the starting point: train the default recipe on every seed to see
its score and how much that score varies by chance (the noise floor); every later change is
measured against it. Then run the research loop on your own: propose one change, run it on
several seeds, let finetune-kit judge it against the noise, record what it taught, and go again,
until the stop rule. Everything is kept in files, so this step can be stopped and resumed at any
time.

## Before you start

{{ORDER_CHECK}}

## How to work in this step

{{HOW_TO_WORK}}

**Research runs unattended.** Never ask the human whether to continue, and do not wait for
replies between runs. The only reasons to stop early are the stop rule, an error you cannot fix
(see below), or the user interrupting you.

## Part 1: the starting point

1. **Resume check.** Run `finetune-kit compare <exp>` and `finetune-kit status <exp>`.
   - If `noise.sigma_obj` is already set, the starting point is measured: if `baseline.md` has
     no *Noise floor* section yet, write it from what `compare` shows (step 4 below), then go to
     Part 2.
   - If a `noise` run is queued or running, wait for it and continue from step 3 below.

2. **Starting point.** Explain: the default recipe is now trained once per seed (the experiment's
   `seeds`), to measure where the model starts and how much its score varies by chance. The
   spread of the scores, σ, is the noise: any later change must beat the best result by clearly
   more than that to count. Run `finetune-kit run <exp> --stage noise`, then
   wait for it with the wait loop.
   - If it ends in `crash` or `timeout`, explain the seeds' `error` and stop; once the cause is
     fixed, the noise run may be started again.

3. **Read the result.** Run `finetune-kit compare <exp>`. Report in plain words:
   - the baseline score: mean ± σ of the objective (the `noise` row in `runs`), and the
     guardrail mean and `sigma_guard` if there is a guardrail;
   - the **keep bar** (`noise.keep_bar`): "a change has to improve <metric> by more than
     <keep_bar> to be kept";
   - time per run (`minutes` of the noise row) and what that means for the research budget
     (`budget_left`): roughly how many runs fit in the hours left. Runs that take more than 3×
     the noise run's training time per seed are stopped as timeouts.
   - If σ is large compared with the gains the user hopes for, say so: small improvements will
     be hard to prove, and more seeds per run would help (that needs a new experiment).

4. **Write the rest of `baseline.md`** in the experiment folder from
   `.finetune-kit/templates/baseline.md` (keep the *Learning-rate scan* section written in Step
   9 of 14 — Choose the learning speed; if it is missing, fill it in from `compare`).

5. Go straight on to Part 2; do not wait for a reply.

## Part 2: research

### Start or resume

1. If `experiments/<exp>/log.md` exists, read it: it holds every earlier idea, its hypothesis,
   result and lesson. Do not repeat an idea that already failed.
2. Run `finetune-kit compare <exp>`. Note `best` (the best run id), `noise.keep_bar`, `streak`
   (non-keeps in a row), `budget_left`, `stop`, and `runs` (every run with its `config`,
   `objective_mean`, `objective_std`, `verdict`, `description`, `round`).
3. Run `finetune-kit status <exp>`. If a research run is still queued or running from an earlier
   session, wait for it first (wait loop), then record its result in `log.md` if its entry has
   no result yet (step 6 of the loop below).
4. If `log.md` does not exist, create it with a short header: experiment, objective metric and
   direction, guardrail, baseline mean ± σ, keep bar, and the tunable settings and losses from
   `search-space.yaml`. If the search-space round is newer than the last round in `log.md`, add a
   line "Round N: opened <settings>".
5. Tell the user once, in two lines: research is running on its own until the stop rule (10
   non-keeps in a row, or the budget is spent), and they can interrupt any time and resume later
   with `{{CMD}}start-experiment`; `{{CMD}}status` shows progress any time.

### The loop

Repeat:

1. **Check.** Run `finetune-kit compare <exp>`. If `stop` is `true`, go to *Summary*.

2. **Pick one change.** Start from the best run's `config` (from `runs`; the noise run's config
   if the best is still the noise run). Change **one idea**: usually one setting, or a couple of
   settings that only make sense together (for example `loss: "focal"` with `focal_gamma`, or
   `llrd` with an adjusted `lr`). Rules:
   - Only names in `search-space.yaml` `tunable`; only losses in its `losses`.
   - The user chose those losses to experiment with: every one in `compare`'s `losses_untried`
     gets a run (with its own settings, for example `focal_gamma`) before research stops. Try
     them early, right after the learning rate. `stop` stays `false` while any are untried and
     budget is left.
   - Never repeat a config that is already in `runs`; never rerun a crashed config unchanged.
   - Pick values yourself, inside sensible ranges; the CLI rejects nonsense.

   How to choose, roughly in this order of payoff: learning rate and layer-wise LR decay
   (`llrd` 0.75–0.9 often helps, usually with a somewhat higher `lr`); then the loss and class
   weighting when labels are imbalanced; then head and pooling; then regularization (dropout,
   EMA, re-init, FGM, R-Drop); then schedule, warmup and epochs. Follow the evidence: if a
   direction helped, try one more step that way; if it hurt, try the other direction once, then
   move on. After a `keep-simpler`, continue from that simpler config.

3. **Record the hypothesis, then run.** Run
   `finetune-kit run <exp> --config '<full config JSON>' --desc "<one-line description>"`.
   The config is the complete set of overrides (best config plus your change), not just the
   change. Right away, append an entry to `log.md`:

   ```
   ## <run_id> · <one-line description>
   Config: `<config JSON>`
   Hypothesis: <one line: why this should help>
   ```

   If `run` prints an error:
   - an invalid value or a setting that is not tunable: pick a valid config and try again (it
     does not count as a run);
   - "budget spent": go to *Summary*;
   - anything about checksums, tampering or changed files: stop, and tell the user that a
     protected file (rules, search space, splits, `experiment.yaml` or the model profile) was changed, so results can no longer be
     trusted until it is restored;
   - anything else: explain it to the user and stop.

4. **Wait** with the wait loop until the status is `done`, `crash` or `timeout`. If the `wait`
   reply has an `error` that mentions tampering, checksums, changed or protected files, or
   "modified after enqueue", stop the loop and tell the user that a protected file or the run
   request was changed, so results can no longer be trusted until it is restored.

5. **Read the verdict** from the `wait` reply (`verdict`: `keep`, `keep-simpler` or `discard`;
   crashes and timeouts are discards). The verdict is final: never override it or argue with
   it. For the numbers, run `finetune-kit compare <exp>` and find the run's row.

6. **Record the result** under the run's entry in `log.md`:

   ```
   Result: <mean ± std> vs best <mean> (Δ <difference>, bar <keep_bar>) → **<verdict>**
   Learned: <one line: what this tells you about this data and model>
   ```

   For a crash or timeout, write the seeds' `error` instead of the numbers, and do not retry
   that config unchanged.

7. Give the user one short line (run, idea, verdict, current best), then go back to step 1 of the loop.

### Summary

When `compare` says `stop: true` (or the budget is spent), append a `## Summary (round N)`
section to `log.md`, where N is the `round` in `search-space.yaml`:
- why research stopped (10 non-keeps in a row, or budget);
- how many runs, and how many keeps, keep-simplers, discards and crashes;
- the chain of kept changes, in order;
- best run vs baseline: mean ± std of each, the difference, and the keep bar;
- what did not help;
- ideas for next time, including settings that are not open yet.

Tell the user in a few lines what research found.

## Finish

{{FINISH}}
