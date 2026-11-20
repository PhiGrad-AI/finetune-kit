# finetune-kit: choose-model

Step 3 of 14. Ask which pre-trained model to start from, or which 2-3 models to compare, and
record the answer.

## Before you start

{{ORDER_CHECK}}

If `set model` says the data is checked and locked, the model can no longer be changed this way:
say so. Before the experiments start, `{{CMD}}compare-models` can still compare models; after
that, a different model needs a new experiment (`{{CMD}}choose-task`).

## How to work in this step

{{HOW_TO_WORK}}

## Steps

1. **List the models.** Run `finetune-kit models --task <task>` (the task is in
   `experiments/<exp>/experiment.yaml`). Show each model on one line: its `key`, the Hugging Face
   id, default learning rate (`lr_default`), max length in tokens (`max_len`) and its `notes` if
   any. Explain in one line: a pre-trained model already knows the language; fine-tuning teaches
   it the user's task. Bigger models are often better but slower.

2. **Ask:** "Which model shall we start from?" Also offer **comparing**: the user picks 2-3
   models, and Step 6 of 14 — Compare models trains each one on their data and keeps the best. Wait for the answer.
   - If they want to compare, ask which 2-3 models (one question). The first one is used until
     the comparison picks the winner.
   - If they want a model that is not listed, run `finetune-kit models add <hf_id>` (add
     `--trust-remote-code` only if the user explicitly says they trust that model's code); such a
     model gets a generic profile, so a learning-rate test (Step 9 of 14 — Choose the
     learning speed) is recommended for it.

3. **Record it.** Run `finetune-kit set model <exp> <model key>`, or to compare,
   `finetune-kit set model <exp> <first> --compare <first>,<b>[,<c>]`. On an error, explain it,
   ask the one question that fixes it, and run it again.

4. **Update `experiment.md`:** fill in the *Model* and *Bake-off* rows with their reasons
   (comparison: no, or yes with the candidates).

## Finish

{{FINISH}}
