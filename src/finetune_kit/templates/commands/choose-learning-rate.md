# finetune-kit: choose-learning-rate

Step 9 of 14. Decide the starting learning rate: the model's usual value, or a short test (an
LR scan) that finds a good one for this data. Run the test if the user wants it. Write the
learning-rate part of `baseline.md`.

## Before you start

{{ORDER_CHECK}}

If `set learning-rate` says the learning rate is fixed once the experiment has started, the
experiments have already begun and the starting learning rate cannot change any more: say so in
plain words and use the hand-off in the Finish section.

## How to work in this step

{{HOW_TO_WORK}}

## Steps

1. **Resume check.** Run `finetune-kit status <exp>` and `finetune-kit compare <exp>`. If an
   `lr-scan` run is queued or running, the test was started before: wait for those runs (step 4).
   If `runs` already has `lr-scan` rows and none is pending, go to step 5.

2. **Explain**, in plain words: the learning rate is how big a step the model takes each time it
   learns from a batch of examples. Too big and training becomes unstable; too small and it
   learns slowly or too little. Every model has a usual value (`lr_default` in
   `finetune-kit models --task <task>`). The test is five short training runs (each about a fifth
   of a full one, one seed) at learning rates spread over the model's safe range; the best one
   becomes the starting value. It costs some time; using the usual value costs none.

   Recommend the test when the model's notes say "generic: lr-scan required" (it has no known
   good value), or when `lr` is in the `tunable` list of `experiments/<exp>/search-space.yaml`
   (the user wants the learning rate to matter); otherwise recommend the usual value. Say why.

3. **Ask:** "Shall I use the model's usual learning speed, or run a short test to find one?"
   Wait for the answer.
   - Usual value: run `finetune-kit set learning-rate <exp> default`, then go to step 6. If the
     model's notes say "generic: lr-scan required", warn first that its usual value is only a
     guess, and go on only if the user still wants it.
   - Test: run `finetune-kit set learning-rate <exp> scan`, then `finetune-kit lr-scan <exp>`. It
     returns `run_ids` and `lrs`.

4. **Wait** for each run id in turn with the wait loop.

5. **Result.** Run `finetune-kit compare <exp>` and show a table: learning rate → objective
   (from the scan runs' `objective`), with crashes marked. The best one (highest, or lowest for
   lower-is-better metrics) becomes the experiment's starting learning rate automatically. Say
   plainly that these are single short runs, so the test only picks a sensible starting point;
   research refines it if `lr` is open. If every scan run crashed, explain the errors in plain
   words and ask whether to use the usual value instead
   (`finetune-kit set learning-rate <exp> default`); otherwise stop.

6. **Write the learning-rate part of `baseline.md`** in the experiment folder from
   `.finetune-kit/templates/baseline.md` (the *Learning-rate scan* section: skipped and why, or
   the table and the chosen value). Step 11 fills in the rest.

## Finish

{{FINISH}}
