# finetune-kit: start

Show the user every step of fine-tuning in plain words, say where they are, and offer the next
step. This step changes nothing.

## Before you start

- `.finetune-kit/memory/research-rules.md` must exist. If it does not, finetune-kit is not
  installed in this project: tell the user to run `finetune-kit init --ai <their agent> --here`,
  and stop.
- Run `finetune-kit version`. If the command is not found, tell the user to install the CLI with
  `uv tool install` (see the finetune-kit README), and stop.

## How to work in this step

{{HOW_TO_WORK}}

## Steps

1. Greet the user in one line: finetune-kit trains a model on their own examples, one small
   decision at a time, and you will guide them through each step and ask before every one.

2. Run `finetune-kit steps` (no argument: it looks at the highest-numbered experiment, if any).
   The reply has `experiment` (the folder name, or `null` when there is none yet), `steps` (each
   with `step`, `title`, `what`, `status`, and a `number` for every step except `start`) and
   `next`.

3. Show the steps as a numbered list, one line each: `number`, `title`, then `what`. Skip the
   `start` row (this is it). Introduce the list with exactly "Here are the steps:", like this:

   ```
   Here are the steps:
    1. Choose the task: Say what kind of problem the model should learn to solve
    2. Choose the data: Point to the examples the model will learn from
    ...
   14. Finish: Train and score the final model, then wrap up
   ```

   Use the `title` and `what` from the reply, not this example. Then say in plain words:
   - steps 1-4 are questions about what they want; step 5 checks their data;
   - step 6 only happens if they choose to compare models;
   - steps 7-10 decide what the experiments may try, and how much;
   - step 11 runs the experiments on their own, often for hours; step 12 (check progress) can be
     used any time while they run; steps 13-14 explain the results and produce the final model.

4. **Where they are.**
   - If `experiment` is `null`: there is no experiment yet, so the first step is Step 1 of 14 —
     Choose the task (`{{CMD}}choose-task`).
   - Otherwise: name the experiment, mark the steps whose `status` is `done` with ✓ in the list
     above (`skipped` steps as "skipped", `optional` ones as "optional"), and say which step is
     next and why (`next.reason`, in plain words). If `next.stage` is `choose-task` with the
     reason "this experiment is complete; start a new one", say the experiment is finished and
     the next step would begin a new one.

## Finish

{{FINISH}}
