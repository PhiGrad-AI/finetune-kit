# Report: <NNN-name>

<Final, or interim if research has not stopped yet.> HTML report: `report.html`.

## Result

| Run | Description | Objective mean ± std | Guardrail mean |
|---|---|---|---|
| <noise run> | baseline (default recipe) | <mean ± std> | <mean> |
| <best run> | <description> | <mean ± std> | <mean> |

<Plain words: the difference, compared with the keep bar.>

## What helped

<Kept changes, in order, with the size of each gain.>

## What hurt or had no effect

<Discarded ideas, grouped, with what they taught. Crashes and timeouts with their cause.>

## Error analysis (best run <run id>, validation data only)

<For token-cls, replace the two sections below with entity errors: missed entities, extra
entities, wrong type, wrong boundaries.>

### Missed positives

<Patterns, with two or three short examples (id and a few words).>

### False alarms

<Patterns, with two or three short examples.>

### Possible label problems (please check)

finetune-kit never changes your data. These rows look mislabelled to the model; fixing them is
your call, and fixed data needs a new experiment.

| id | Label | Model says | Confidence |
|---|---|---|---|
| <id> | <label> | <predicted> | <score> |

## What to try next

- Untried settings that were open: <names>
- Settings worth opening in a new round: <names and why>
- Data fixes: <from the error analysis>

## Next

Promote the best run: retrain on all training data and score the test set once.
