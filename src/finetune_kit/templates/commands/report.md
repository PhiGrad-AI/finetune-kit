# finetune-kit: report

Step 13 of 14. Explain what the research found: the best recipe against the baseline, what
helped and what hurt, what to open next, and an error analysis of the best run's validation
mistakes. Write `report.md`; finetune-kit writes `report.html`.

## Before you start

{{ORDER_CHECK}}

**Interim report.** If `next` names `start-experiment` and the user declines it but asks for a
report anyway, you may go on: say plainly that this report is interim, because research round N (the
`round` in `search-space.yaml`) has no `## Summary (round N)` in `log.md` yet (a bare `## Summary`
counts for round 1). If `log.md` is missing, take what helped and hurt from the verdicts in
`finetune-kit compare` instead.

## How to work in this step

{{HOW_TO_WORK}}

## Steps

1. **Numbers.** Run `finetune-kit compare <exp>`. Find the best run (`best`) and the noise run
   (`noise.noise_run`) in `runs`.

2. **HTML report.** Run `finetune-kit report <exp>`. It prints the `path` of `report.html`.

3. **Mistakes.** Run `finetune-kit misses <exp> <best run id>`. It reads validation data only.
   - seq-cls: `false_negatives` (positives the model missed, lowest scores first),
     `false_positives` (false alarms, highest scores first), `possible_label_issues` (rows where
     the model is very sure the label is something else: `label` vs `predicted`, with `score`),
     and the `threshold` used, if any.
   - token-cls: `rows`, each with the gold entities (`gold`) and predicted ones (`pred`).

   Look for **patterns**, not single rows: length, topic, wording, formatting, a label or entity
   type that keeps being confused, entity boundaries that are off by a word, and so on.

4. **Untried settings.** Compare `search-space.yaml` `tunable` with the configs in `runs`:
   settings that were open but never changed are untried. Run `finetune-kit catalog <exp>` and
   note ★ settings that are not open yet.

5. **Write `report.md`** in the experiment folder from `.finetune-kit/templates/report.md`:
   - **Result:** best run vs baseline, mean ± std for each, the difference and the keep bar, in
     plain words ("about 0.04 better, roughly ten times the noise").
   - **What helped / what hurt / no effect:** from the verdicts and `log.md` lessons, with the
     size of each effect.
   - **Error analysis:** the patterns in misses and false alarms (or entity errors), with two or
     three short examples each (id and a few words, not whole texts). List **every**
     `possible_label_issues` row as a likely labelling problem for the user to check (id, its
     label, what the model predicts, confidence). Say plainly that finetune-kit never changes
     their data; fixing labels is their call, and fixed data needs a new experiment.
   - **What to try next:** untried settings, ★ settings and other losses to open in another
     round (`{{CMD}}choose-loss`, then `{{CMD}}choose-hyperparameters`), and data fixes suggested
     by the error analysis.

6. Tell the user the main findings in a few lines, and the paths of `report.md` and
   `report.html`. Keep the example texts in the project; do not copy them anywhere else.

## Finish

`finetune-kit next` offers two ways on: finish (train the final model and score the test set
once), or another round with other losses or settings (`choose-loss`). Show both.

{{FINISH}}
