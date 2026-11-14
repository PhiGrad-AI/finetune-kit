# Experiment <NNN-name>

Created <date> with finetune-kit <version>.

## Goal

<One or two sentences in the user's own words: what the model should do, and what a mistake
costs them.>

## Choices

| Setting | Value | Why |
|---|---|---|
| Task | <seq-cls or token-cls> | <reason> |
| Data | <one file split 80/10/10 by finetune-kit, or train / val / test paths> | <reason> |
| Model | <model key> | <reason> |
| Bake-off | <no, or yes: candidates a, b, c (the first is a placeholder until the bake-off)> | <reason> |
| Objective metric | <metric, its target if any, higher or lower is better> | <reason> |
| Positive class | <label, or not needed> | <reason> |
| Guardrail metric | <metric and target, or none> | <reason> |
| Seeds per run | <N> | <reason> |
| Budget | <max runs> runs, <max hours> hours | <reason> |
| Research subset | <rows, or full training set> | <reason> |

## Notes

<Anything the user said that matters later: data quirks, deadlines, how the model will be used.>

## Next

Check the data and freeze the splits (the check stage).
