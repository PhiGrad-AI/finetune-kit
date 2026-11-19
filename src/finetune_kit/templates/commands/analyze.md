# finetune-kit: analyze

Explain what the research found: the best recipe against the baseline, what helped and what
hurt, what to open next, and an error analysis of the best run's validation mistakes. Write
`report.md`; finetune-kit writes `report.html`.

## Before you start

`experiments/<exp>/log.md` must exist. If it does not, the research stage has not run:
{{GOTO_NEXT}}

If `log.md` has no `## Summary` yet, research is
still going or was interrupted: say this report is interim, and go on.

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
   - **What to try next:** untried settings, ★ settings to open with `{{CMD}}hparams`, and data
     fixes suggested by the error analysis.

6. Tell the user the main findings in a few lines, and the paths of `report.md` and
   `report.html`. Keep the example texts in the project; do not copy them anywhere else.

## Finish

{{FINISH}}
