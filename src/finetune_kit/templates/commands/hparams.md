# finetune-kit: hparams

Show what can be tuned for this task and model, with recommendations, let the user say which
settings the research may change (names only; you pick the values later), commit that search
space, and write `search-space.md`.

## Before you start

{{ORDER_CHECK}}

**New round?** If `experiments/<exp>/search-space.yaml` already exists, this is a rerun. Read its
`round` field and tell the user: "This starts round N+1. The best run so far, the noise floor (σ)
and the remaining budget carry over; research continues from the current best with the new
settings open."

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

1. Run `finetune-kit catalog <exp>`. The reply has two lists:
   - `catalog`: the settings. Each has `name`, `group`, `default`, `help`, `recommended` and
     `reason`.
   - `losses`: the training losses this task can use. Each has `name`, `help`, `always` (true for
     `ce`, which is always allowed), `recommended` and `reason`.

   Settings and losses are two separate questions below. Ask them one at a time.

2. **Present the settings grouped**, like this (★ = recommended, with the reason in a few words;
   the rest on one line per group, names with a short plain explanation). Leave out the `loss`
   entry itself: losses get their own question in step 4.

   ```
   For seq-cls on <model>, here is what can be tuned. ★ = recommended for your data.
     Optimizer & schedule
      ★ lr               learning rate: most sensitive setting
      ★ llrd             layer-wise LR decay: biggest single gain on earlier projects
        warmup · schedule · weight_decay · epochs · batch · head_lr_mult · clip · beta2
     Head:  pooling · head · ...
     Loss settings
      ★ class_weight     labels are 18% / 82%
        label_smoothing · ...
     Regularization: dropout · ema · reinit_top · freeze_embed · fgm_eps · swa · ...
     Speed: precision (bf16) · max_len
   ```

   Use the groups from the catalog; keep it compact. Settings that belong to one loss
   (`focal_gamma`, `asl_*`, `rank_weight`, `dice_*`) are not shown here; they come up in step 5.
   If the round is not the first, mark which settings are already open.

3. **Ask about settings:** "Which of these may I change? Name them; anything left out stays at
   its default. You can also say 'the starred ones'." Suggest keeping the first round short (the
   starred ones plus a few) because every open setting costs runs; more can be opened later by
   rerunning `{{CMD}}hparams`. Learning rate (`lr`) being open means the baseline stage runs an LR
   scan. Wait for the answer.

4. **Ask about losses.** Show every entry of the `losses` list from step 1, one line each, with ★
   and the reason for recommended ones, like this:

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
   the round is not the first, say which losses are already allowed. For token classification,
   `dice` cannot be combined with a `crf` head; say so if `crf` is in play.

5. **Loss settings.** For each loss the user allowed besides `ce`, offer its own settings in one
   line (`focal` → `focal_gamma`; `asl` → `asl_gamma_neg`, `asl_gamma_pos`, `asl_clip`; `rank` and
   `ce+rank` → `rank_weight`; `dice` → `dice_smooth`, `dice_ce_weight`) and ask whether research
   may change them too, or keep their defaults. Skip this step when only `ce` was chosen.

6. **Confirm** both lists in one message: the settings research may change, and the losses it may
   try. If any loss besides `ce` is allowed, say that `loss` itself becomes tunable so research
   can switch between them. Ask: "Commit this search space?"

7. **Commit:** `finetune-kit space commit <exp> --tunable <a,b,c> --losses <ce,focal>`. `--losses`
   is required, even when it is just `ce`. The reply shows `round`, `tunable` (with `loss` added
   when a second loss is allowed) and `losses`. On an error (for example an unknown name),
   explain it, correct the list with the user, and commit again.

8. **Write `search-space.md`** in the experiment folder from
   `.finetune-kit/templates/search-space.md`: the round, each open setting and why (★ reason or
   the user's reason), the allowed losses, and what stays fixed. On a new round, keep the earlier
   rounds' history in the file and add the new round below it.

## Finish

{{FINISH}}
