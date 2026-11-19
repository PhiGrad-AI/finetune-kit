# finetune-kit: research

Run the research loop on your own: propose one change, run it on several seeds, let
finetune-kit judge it against the noise, record what it taught, and go again, until the stop
rule. Everything is kept in files, so this stage can be stopped and resumed at any time.

## Before you start

`experiments/<exp>/baseline.md` must exist. If it does not:

{{GOTO_NEXT}}

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

**This stage runs unattended.** Never ask the human whether to continue, and do not wait for
replies between runs. The only reasons to stop early are the stop rule, an error you cannot fix
(see below), or the user interrupting you.

## Start or resume

1. If `experiments/<exp>/log.md` exists, read it: it holds every earlier idea, its hypothesis,
   result and lesson. Do not repeat an idea that already failed.
2. Run `finetune-kit compare <exp>`. Note `best` (the best run id), `noise.keep_bar`, `streak`
   (non-keeps in a row), `budget_left`, `stop`, and `runs` (every run with its `config`,
   `objective_mean`, `objective_std`, `verdict`, `description`, `round`).
3. Run `finetune-kit status <exp>`. If a research run is still queued or running from an earlier
   session, wait for it first (wait loop), then record its result in `log.md` if its entry has
   no result yet (step 6 below).
4. If `log.md` does not exist, create it with a short header: experiment, objective metric and
   direction, guardrail, baseline mean ± σ, keep bar, and the tunable settings and losses from
   `search-space.yaml`. If the search-space round is newer than the last round in `log.md`, add a
   line "Round N: opened <settings>".
5. Tell the user once, in two lines: research is running on its own until the stop rule (10
   non-keeps in a row, or the budget is spent), and they can interrupt any time and resume later
   with `{{CMD}}research`.

## The loop

Repeat:

1. **Check.** Run `finetune-kit compare <exp>`. If `stop` is `true`, go to *Finish*.

2. **Pick one change.** Start from the best run's `config` (from `runs`; the noise run's config
   if the best is still the noise run). Change **one idea**: usually one setting, or a couple of
   settings that only make sense together (for example `loss: "focal"` with `focal_gamma`, or
   `llrd` with an adjusted `lr`). Rules:
   - Only names in `search-space.yaml` `tunable`; only losses in its `losses`.
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
   - "budget spent": go to *Finish*;
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

7. Give the user one short line (run, idea, verdict, current best), then go back to step 1.

## Finish

When `compare` says `stop: true` (or the budget is spent), append a `## Summary (round N)` section to
`log.md`, where N is the `round` in `search-space.yaml`:
- why research stopped (10 non-keeps in a row, or budget);
- how many runs, and how many keeps, keep-simplers, discards and crashes;
- the chain of kept changes, in order;
- best run vs baseline: mean ± std of each, the difference, and the keep bar;
- what did not help;
- ideas for next time, including settings that are not open yet.

Tell the user in a few lines what research found.

{{FINISH}}
