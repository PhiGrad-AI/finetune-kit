# finetune-kit: setup

Create a new fine-tuning experiment from the user's answers, and leave
`experiments/NNN-<name>/experiment.md` explaining every choice and why.

## Before you start

- `.finetune-kit/memory/research-rules.md` must exist. If it does not, finetune-kit is not
  installed in this project: tell the user to run `finetune-kit init --ai <their agent> --here`,
  and stop.
- Run `finetune-kit version`. If the command is not found, tell the user to install the CLI with
  `uv tool install` (see the finetune-kit README), and stop.

## How to work in this stage

- First read `.finetune-kit/memory/research-rules.md` and follow it throughout.
- Ask **one question at a time** and wait for the answer. Use plain language and explain any
  ML term in a few words the first time it comes up.
- Act only through `finetune-kit …` commands. Each prints one JSON object: read it before you
  go on, and base what you tell the user on it, not on guesses.
- Never edit files that finetune-kit writes (`experiment.yaml`, `search-space.yaml`,
  `state.json`, `results.tsv`, `data-report.json`, `splits/`, `runs/`, `final/`,
  `.finetune-kit/memory/`). Never open anything under `~/.finetune-kit/` or
  `experiments/*/splits/`, and do not open the user's dataset files.
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

(This stage creates the experiment, so the "which experiment" rule applies only once
`finetune-kit new` has created it.)

## Steps

Greet the user in one line: you will ask a few questions, one at a time, then create the
experiment. Then ask, in this order:

1. **Task.** Ask which of these fits, one line each:
   - **Sequence classification** (`seq-cls`): one label per text, for example spam / not spam,
     or which department an email belongs to.
   - **Token classification** (`token-cls`): a tag for each word, for example finding names,
     addresses or other entities in text (NER).

   Then tell them the data format it needs, as JSONL (one JSON object per line):
   - seq-cls: `{"id": ..., "text": "...", "label": ...}` (label is a number or a string)
   - token-cls: `{"id": ..., "tokens": ["...", ...], "tags": ["B-PER", "O", ...]}` (BIO or BIOES tags)
   - optional `"group_id"` on every row (for example the source document) keeps related rows in
     the same split, so near-copies cannot leak from training into validation.

2. **Data.** Ask for the path to the data: either one file (finetune-kit splits it 80/10/10
   into train / validation / test) or three files (train, validation, test; give both validation
   and test, or neither). Explain in one line that the test part is locked away and scored only
   once, at the very end. You may check that the files exist (`ls`) and count rows
   (`wc -l <file>`), but do not open them.

3. **Model.** Run `finetune-kit models --task <task>`. Show each model on one line: its `key`,
   the Hugging Face id, default learning rate (`lr_default`), max length in tokens (`max_len`)
   and its `notes` if any. Offer a fourth option, **"bake-off"**: the user picks 2–3 candidates
   and a later stage trains each one and picks the best. If they choose a bake-off, ask which 2–3
   models; the first one is used as a placeholder until the bake-off stage fixes the model.
   If they want a model that is not listed, run `finetune-kit models add <hf_id>` (add
   `--trust-remote-code` only if the user explicitly says they trust that model's code); such a
   model gets a generic profile and needs a learning-rate scan at the baseline stage.

4. **Metric to optimize.** Run `finetune-kit metrics --task <task>` and list the metrics, one
   plain line each. Use these meanings:
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

   Help them choose by asking what a mistake costs them, if they are unsure. If the metric has a
   recall or precision target in `params` (`recall_min`, `recall`, `precision`), ask whether the
   default target suits them; leave any other params (such as `grid`) at their defaults. If the
   metric has `needs_positive: true`, ask which label is the positive class (the one they care
   about catching); pass it as `--positive-class`.

5. **Guardrail (optional).** Ask whether there is a second metric that must not get worse while
   the first one improves (for example optimize `pauc_hi` but protect `skip_at_recall`). Default:
   none. Ask about its target and positive class the same way.

6. **Seeds per run.** Explain: every run trains the same recipe several times with different
   random seeds and averages the results, because one training run alone can be lucky. More
   seeds give more trustworthy comparisons but take longer. Default **3**; at least 2.

7. **Budget.** Ask how many research runs (default 40) and how many hours (default 24) the
   research may use at most.

8. **Research subset.** Only if the training data has more than 5,000 rows: offer to run the
   research on a random subset (for example 3,000 rows) to make each run faster. Explain that the
   final model is still retrained on all the data at the end. Skip this question otherwise.

9. **Confirm.** Propose a short experiment name (lowercase, hyphens, e.g. `demo-exp`) and show
   all choices as a short list. Ask: "Shall I create the experiment?"

10. **Create.** Run:

    ```
    finetune-kit new <name> --task <task> --model <model key> --train <path>
        [--val <path> --test <path>] --metric <metric> [--metric-params '<json>']
        [--guardrail <metric> --guardrail-params '<json>'] [--positive-class <label>]
        [--custom-metric <path:func> --direction max|min]
        --seeds <N> --max-runs <runs> --max-hours <hours> [--subset <rows>]
    ```

    `--metric-params` takes only the params the user changed, e.g. `'{"recall": 0.99}'`.
    The JSON reply has `experiment` (the folder name) and `path`. If it is an error (for example
    a label that is not in the data, or a positive class that is needed), explain it, ask the one
    question that fixes it, and run `new` again.

11. **Write `experiment.md`** in the new experiment folder, from
    `.finetune-kit/templates/experiment.md`. Fill in every section in plain words: what the
    user wants, each choice and why it was made. Record clearly whether a bake-off was chosen
    and which candidates, because the bake-off stage reads that.

## Finish

Tell the user the experiment folder name and that the next step checks the data and freezes
the splits. Next command: `{{CMD}}check`.
