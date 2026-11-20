# finetune-kit: choose-task

Step 1 of 14. Ask what kind of problem the model should learn, tell the user the data format it
needs, and create the experiment with only that answer. Start `experiment.md`.

## Before you start

- `.finetune-kit/memory/research-rules.md` must exist. If it does not, finetune-kit is not
  installed in this project: tell the user to run `finetune-kit init --ai <their agent> --here`,
  and stop.
- Run `finetune-kit version`. If the command is not found, tell the user to install the CLI with
  `uv tool install` (see the finetune-kit README), and stop.

{{ORDER_CHECK}}

**An experiment in progress.** This step always creates a new experiment. If `next` names any
step other than `choose-task`, the experiment it names (`experiment`) is still in progress. Say so
in plain words (which step it is at) and ask whether they want to carry on with it (show its next
step the way the order check above shows one) or start a new experiment. Only go on here if
they want a new one.

## How to work in this step

{{HOW_TO_WORK}}

(This step creates the experiment, so the "which experiment" rule applies only once
`finetune-kit new` has created it.)

## Steps

1. **Task.** Ask: "Which of these fits what you want the model to do?" Offer, one line each:
   - **Label whole texts** (sequence classification, `seq-cls`): one label per text, for
     example spam / not spam, or which department an email belongs to.
   - **Tag words in a text** (token classification, `token-cls`): a tag for each word, for
     example finding names, addresses or other entities in text (NER).

   Wait for the answer.

2. **Data format.** Tell them the data format it needs (they choose the data itself in the next
   step). Files can be JSONL (one JSON object per line, shown below), CSV, TSV or Parquet with a
   column for each field; the column names can be mapped later:
   - seq-cls: `{"id": ..., "text": "...", "label": ...}` (label is a number or a string)
   - token-cls: `{"id": ..., "tokens": ["...", ...], "tags": ["B-PER", "O", ...]}` (BIO or BIOES tags)
   - optional `"group_id"` on every row (for example the source document) keeps related rows in
     the same split, so near-copies cannot leak from training into validation.

3. **Name and create.** Propose a short experiment name from what they told you (lowercase,
   hyphens, e.g. `support-tickets`) and ask whether it is fine. Then run
   `finetune-kit new <name> --task <seq-cls|token-cls>`. The JSON reply has `experiment` (the
   folder name) and `path`; `<exp>` is its number from now on. The data, model and goal are
   answered in the next steps.

4. **Start `experiment.md`** in the new experiment folder, from
   `.finetune-kit/templates/experiment.md`: fill in *Goal* in the user's own words and the
   *Task* row with its reason. Leave the other rows as they are; the next steps fill them in.

## Finish

Tell the user the experiment folder name.

{{FINISH}}
