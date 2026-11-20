# finetune-kit: plain-language guided steps

Design, follow-up to `2026-11-14-finetune-kit-design.md` and `2026-11-19-stage-handoff-design.md`.

## Why

The eight stages (setup, check, bakeoff, hparams, baseline, research, analyze, promote) bundle
several decisions each and use engineer names. The user wants a flow a non-engineer can follow:
one decision per step, names that say what the step does, a summary of all steps at the start,
and guidance to the next step after each one (the hand-off from PR #4 stays).

## The steps

| # | Command (Codex `$`, Claude Code `/`, Cursor `/finetune-kit.<step>`) | What the user decides or sees |
|---|---|---|
| 0 | `finetune-kit-start` | Summary of all steps; then begins step 1 |
| 1 | `finetune-kit-choose-task` | Classify whole texts, or tag words in them |
| 2 | `finetune-kit-choose-data` | One file, three files, or a Hugging Face dataset; which column is which |
| 3 | `finetune-kit-choose-model` | One model, or 2–3 to compare |
| 4 | `finetune-kit-choose-goal` | The score to improve, and an optional "must not get worse" score |
| 5 | `finetune-kit-check-data` | Validate and split the data; test part locked away |
| 6 | `finetune-kit-compare-models` | Only if 2–3 models were chosen: train each, pick one |
| 7 | `finetune-kit-choose-loss` | Which losses research may try (★ recommended) |
| 8 | `finetune-kit-choose-hyperparameters` | Which settings research may tune (★ recommended) |
| 9 | `finetune-kit-choose-learning-rate` | The model's default, or a short scan to find one |
| 10 | `finetune-kit-choose-experiments` | How many experiments and hours research may use, seeds per experiment, optional smaller research subset |
| 11 | `finetune-kit-start-experiment` | Measure the noise floor, then research runs on its own |
| 12 | `finetune-kit-status` | Progress any time: experiments done, best score so far, budget left |
| 13 | `finetune-kit-report` | What helped, what didn't, the mistakes the model makes |
| 14 | `finetune-kit-finish` | Final retrain, the one-time test score, `predict.py` |

Changes from the user's list, and why:
- **Model, goal and experiment budget are their own steps**: the user asked to be told of
  missing steps, and named "choose experiments" before "start experiment".
- **Check data comes after model and goal** (not straight after choose-data): checking measures
  text length with the chosen model's tokenizer and warns when the validation set is too small
  for the chosen score. It still comes before loss, hyperparameters and learning rate, whose
  recommendations are based on what the check finds.
- **Compare models is a separate optional step after the check**, because comparing means
  training, which needs the checked, split data.
- **Status and report can be run any time** once research has started; the hand-off order is
  start-experiment → (status while running) → report → finish.

## How it works

**The experiment is created at step 1 and filled in step by step.** `finetune-kit new NAME
--task T` creates `experiments/NNN-name/` with only the task. A new `finetune-kit set <exp>
<field> …` records each later answer (data source/columns, model or candidates, goal, budget,
learning-rate choice). Each answer is validated as it is given (same checks as today's `new`).
Answers can be changed until the data is checked; after that, data/model/goal are locked, and
the budget until research starts (today's checksum protection applies from then on). The
one-shot `new` with all options keeps working for scripts and old experiments.

**`finetune-kit next` follows the new order**, using which fields are set and the existing state
(frozen splits, model chosen, losses and search space committed, learning rate decided, budget
set, noise floor, research summary, report, promoted). Its stage names are the new step names.
`finetune-kit steps` prints the full list with a one-line description each and which are done —
used by `start` and by every hand-off ("Step 7 of 14").

**Losses and hyperparameters are committed separately**: `space commit` is split into
`finetune-kit set <exp> losses …` (step 7) and `space commit --tunable …` (step 8, which keeps
the committed losses). Learning rate: `set <exp> learning-rate default|scan`; `scan` runs today's
LR scan in that step.

**Stage prompts** are rewritten one per step, each asking only its own question(s), with the
shared start check and the shared "Next: … Command: … Shall I start it now?" ending. `start`
shows the summary (from `finetune-kit steps`) and offers step 1, or, if an experiment is in
progress, says where the user is and offers the next step. `research-rules.md` and README are
updated to the new names.

**Old names:** existing installs and experiments keep working — `init` removes the old eight
prompt files it wrote and installs the new ones; old experiments (all fields already set) map
onto the new `next` order.

## Non-goals
- Changing what training, research or promotion do.
- Changing the keep rule, budgets' meaning, or the test-vault guarantee.

## Left to the builder
- Exact `set` field names and messages; the wording of each step prompt.
