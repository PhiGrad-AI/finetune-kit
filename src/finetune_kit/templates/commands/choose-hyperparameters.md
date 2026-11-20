# finetune-kit: choose-hyperparameters

Step 8 of 14. Show which training settings (hyperparameters) can be tuned for this task and
model, with recommendations, let the user say which ones research may change (names only; you
pick the values later), commit that search space with the losses chosen in Step 7 of 14 — Choose
the loss, and write `search-space.md`.

## Before you start

{{ORDER_CHECK}}

**A new round always starts with the loss question.** Run `finetune-kit compare <exp>`. If the
experiments have already started (it shows a `noise.sigma_obj`):
- If `next.stage` from the order check is `choose-hyperparameters`, a new round is open (the
  user chose its losses in Step 7 of 14 — Choose the loss). Read the `round` in
  `experiments/<exp>/search-space.yaml` (reading is fine) and say the settings chosen now apply
  from round N+1. Go on below.
- Otherwise no new round is open. Do not ask about settings and do not commit anything here:
  committing now would start a new round with the old losses and skip the loss question. Tell
  the user that more settings are opened in a new round, which starts at Step 7 of 14 — Choose
  the loss and then comes back to this step. Show it:
  `Next: Step 7 of 14 — Choose the loss (a new round starts with the loss question). Command: {{CMD}}choose-loss`.
  Ask "Shall I start Choose the loss now?" and wait. If the user says yes, open
  `{{STEP_FILE}}`
  (with `<step>` replaced by `choose-loss`) and follow it from the top. If no, stop.

## How to work in this step

{{HOW_TO_WORK}}

## Steps

1. Run `finetune-kit catalog <exp>`. Its `catalog` list holds the settings; each has `name`,
   `group`, `default`, `help`, `recommended` and `reason`. Its `losses` list was the question of
   Step 7 of 14 — Choose the loss (`{{CMD}}choose-loss`); the losses chosen there are used here.

2. **Present the settings grouped**, like this (★ = recommended, with the reason in a few words;
   the rest on one line per group, names with a short plain explanation). Leave out the `loss`
   entry itself: losses were chosen in the step before.

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
   (`focal_gamma`, `asl_*`, `rank_weight`, `dice_*`) are not shown here; they come up under *Loss settings* below.
   If the round is not the first, mark which settings are already open.

3. **Ask about settings:** "Which of these may I change? Name them; anything left out stays at
   its default. You can also say 'the starred ones'." Suggest keeping the first round short (the
   starred ones plus a few) because every open setting costs runs; more can be opened later in
   another round (`{{CMD}}choose-loss`, then this step). Learning rate (`lr`) being open means research
   may change it; the next step decides its starting value. Wait for the answer.

4. **Loss settings.** For each loss the user allowed besides `ce`, offer its own settings in one
   line (`focal` → `focal_gamma`; `asl` → `asl_gamma_neg`, `asl_gamma_pos`, `asl_clip`; `rank` and
   `ce+rank` → `rank_weight`; `dice` → `dice_smooth`, `dice_ce_weight`) and ask whether research
   may change them too, or keep their defaults. Wait for the answer. Skip this when only `ce` was
   allowed.

5. **Confirm** in one message: the settings research may change, and the losses chosen in
   Step 7 of 14 — Choose the loss.
   If any loss besides `ce` is allowed, say that `loss` itself becomes tunable so research can
   switch between them. Ask: "Commit this search space?" and wait.

6. **Commit:** `finetune-kit space commit <exp> --tunable <a,b,c>`. The losses chosen in
   Step 7 of 14 — Choose the loss are used. If it says `--losses is required`, no losses were
   chosen yet: offer `{{CMD}}choose-loss` first. The reply shows `round`, `tunable` (with `loss` added when a
   second loss is allowed) and `losses`. On an error (for example an unknown name), explain it,
   correct the list with the user, and commit again.

7. **Write `search-space.md`** in the experiment folder from
   `.finetune-kit/templates/search-space.md`: the round, each open setting and why (★ reason or
   the user's reason), the allowed losses, and what stays fixed. On a new round, keep the earlier
   rounds' history in the file and add the new round below it.

## Finish

{{FINISH}}
