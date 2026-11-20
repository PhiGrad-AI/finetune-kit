# finetune-kit: choose-loss

Step 7 of 14. Ask which losses research may try and record them. A loss is how training measures
the model's mistakes; some losses suit imbalanced data or "never miss a positive" goals better
than the standard one.

## Before you start

{{ORDER_CHECK}}

**A new round?** If the experiments have already started (Step 11 of 14 — Run the experiments has
run; `finetune-kit compare <exp>` shows a `noise.sigma_obj`), choosing losses now starts a new round of research. Read the
`round` in `experiments/<exp>/search-space.yaml` (reading is fine) and tell the user before you
ask anything: "This starts round N+1. The best run so far, the starting-point measurement and the
remaining budget carry over; research continues from the current best with the new choices. In
the next step you choose the settings for this round." Go on only if they want a new round.

## How to work in this step

{{HOW_TO_WORK}}

## Steps

1. Run `finetune-kit catalog <exp>`. Use its `losses` list: the training losses this task can
   use. Each has `name`, `help`, `always` (true for `ce`, which is always allowed),
   `recommended` and `reason`. (The `catalog` list of settings is for the next step.)

2. **Ask about losses.** Show every entry of the `losses` list, one line each, with ★ and the
   reason for recommended ones, like this:

   ```
   Which losses may research try? ★ = recommended for your data.
       ce        standard cross-entropy (always allowed)
     ★ focal     focuses on hard examples: minority class is 18% of rows
     ★ asl       pushes harder on missed positives
       ce+rank   cross-entropy plus ranking: pauc_hi rewards ranking quality
       rank      rewards ranking positives above negatives
   Say e.g. "ce, focal", or "the starred ones". Just "ce" keeps the standard loss.
   ```

   Ask: "Which losses may research try?" and wait for the answer. `ce` is always included. If
   this is a new round, say which losses were allowed before. For token classification, `dice`
   cannot be combined with a `crf` head; say so if `crf` is in play.

3. **Record it.** Run `finetune-kit set losses <exp> <ce,focal>`. The reply shows `losses`. If
   any loss besides `ce` is allowed, say that research may switch between them. On an error (for
   example an unknown name), explain it, correct the list with the user, and run it again.

4. **Note it** for `search-space.md`: the next step writes that file with the allowed losses and
   why (★ reason or the user's reason).

## Finish

{{FINISH}}
