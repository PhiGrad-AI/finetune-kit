# finetune-kit: choose-goal

Step 4 of 14. Ask what counts as a good result: the score to improve, its target if it has one,
the label the user cares about most, and an optional score that must not get worse. Record it.

## Before you start

{{ORDER_CHECK}}

If `set goal` says the data is checked and locked, the goal can no longer change for this
experiment: say so, and that changing it needs a new experiment (`{{CMD}}choose-task`).

## How to work in this step

{{HOW_TO_WORK}}

## Steps

1. **List the scores.** Run `finetune-kit metrics --task <task>` and list the metrics, one plain
   line each. Use these meanings:
   - `accuracy`: share of rows labelled correctly.
   - `f1`: balance of precision and recall for the positive class.
   - `macro_f1`: F1 averaged over classes, each class counting equally (good when classes are
     imbalanced). `micro_f1`: F1 over all rows pooled.
   - `roc_auc`: how well scores rank positives above negatives. `pr_auc`: the same idea, focused
     on the positive class (better for rare positives).
   - `log_loss`: how well-calibrated the probabilities are (lower is better).
   - `pauc_hi`: for "never miss a positive" cases: the share of negatives the model can skip
     while keeping recall in a high band (default 98–100%).
   - `skip_at_recall`: share of negatives skipped at a fixed recall (default 99.5%).
   - `precision_at_recall`: precision at a fixed recall (default 99.5%).
   - `recall_at_precision`: recall at a fixed precision (default 95%), for "only flag what you
     are sure of" cases.
   - `entity_f1`: F1 over whole entities (token-cls). `token_accuracy`: share of tokens tagged
     correctly (token-cls).
   - A custom metric: a Python function `metric(y_true, probs) -> float` in a file of theirs
     (seq-cls only); pass `--custom-metric path/to/file.py:func --direction max|min`.

2. **Ask:** "Which score should the model improve?" If they are unsure, help by asking what a
   mistake costs them (a missed case, or a false alarm). Wait for the answer.
   - If the metric has a recall or precision target in `params` (`recall_min`, `recall`,
     `precision`), ask whether the default target suits them; leave any other params (such as
     `grid`) at their defaults.
   - If the metric has `needs_positive: true`, ask which label is the positive class (the one
     they care about catching).

3. **Guardrail (optional).** Ask: "Is there a second score that must not get worse while the
   first one improves?" (for example improve `pauc_hi` but protect `skip_at_recall`). Default:
   none. Ask about its target and positive class the same way.

4. **Record it.** Run:

   ```
   finetune-kit set goal <exp> --metric <metric> [--metric-params '<json>']
       [--guardrail <metric> --guardrail-params '<json>'] [--positive-class <label>]
       [--custom-metric <path:func> --direction max|min]
   ```

   `--metric-params` takes only the params the user changed, e.g. `'{"recall": 0.99}'`. If it is
   an error (for example a label that is not in the data, or a positive class that is needed),
   explain it, ask the one question that fixes it, and run `set goal` again.

5. **Update `experiment.md`:** fill in the *Objective metric*, *Positive class* and *Guardrail
   metric* rows with their reasons.

## Finish

{{FINISH}}
