# finetune-kit: hparams

Show what can be tuned for this task and model, with recommendations, let the user say which
settings the research may change (names only; you pick the values later), commit that search
space, and write `search-space.md`.

## Before you start

- `experiments/<exp>/data-report.md` must exist. If not, tell the user to run
  `{{CMD}}check` first, and stop.
- If `experiment.md` says a bake-off was chosen, `experiments/<exp>/bakeoff.md` must exist too.
  If not, tell the user to run `{{CMD}}bakeoff` first, and stop.
- **New round?** If `experiments/<exp>/search-space.md` already exists, this is a rerun. Read
  `experiments/<exp>/search-space.yaml` (its `round` field) and tell the user: "This starts
  round N+1. The best run so far, the noise floor (σ) and the remaining budget carry over;
  research continues from the current best with the new settings open."

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

1. Run `finetune-kit catalog <exp>`. Each entry has `name`, `group`, `default`, `help`,
   `recommended` and `reason`.

2. **Present it grouped**, like this (★ = recommended, with the reason in a few words; the rest
   on one line per group, names with a short plain explanation):

   ```
   For seq-cls on <model>, here is what can be tuned. ★ = recommended for your data.
     Optimizer & schedule
      ★ lr               learning rate: most sensitive setting
      ★ llrd             layer-wise LR decay: biggest single gain on earlier projects
        warmup · schedule · weight_decay · epochs · batch · head_lr_mult · clip · beta2
     Head:  pooling · head · ...
     Loss
      ★ class_weight     labels are 18% / 82%
      ★ loss             focal: minority class is 18% of rows
        ...
     Regularization: dropout · ema · reinit_top · freeze_embed · fgm_eps · swa · ...
     Speed: precision (bf16) · max_len
   ```

   Use the groups from the catalog; keep it compact. If the round is not the first, mark which
   settings are already open.

3. **Losses.** The `loss` entry chooses the training loss. Explain the options for this task in
   one line each:
   - `ce`: standard cross-entropy (the default; always allowed).
   - `focal`: focuses training on hard examples; helps with imbalanced labels.
   - `asl` (seq-cls): asymmetric loss; pushes harder on missed positives.
   - `rank` / `ce+rank` (seq-cls): rewards ranking positives above negatives; suits AUC-style
     metrics.
   - `dice` (token-cls): overlap-based; helps when entities are rare. Not with `crf`.

   Settings such as `focal_gamma`, `asl_*`, `rank_weight` and `dice_*` only matter if their loss
   is allowed.

4. **Ask:** "Which of these may I change? Name them; anything left out stays at its default.
   You can also say 'the starred ones'." Suggest keeping the first round short (the starred ones
   plus a few) because every open setting costs runs; more can be opened later by rerunning
   `{{CMD}}hparams`. Learning rate (`lr`) being open means the baseline stage runs an LR scan.

5. **Confirm** the final list in one message: the tunable names and the allowed losses. If the
   user allowed any loss besides `ce`, add `loss` to the tunable names and say so. Ask: "Commit
   this search space?"

6. **Commit:** `finetune-kit space commit <exp> --tunable <a,b,c> --losses <ce,focal>`. The reply
   shows `round`, `tunable` and `losses`. On an error (for example an unknown name), explain it,
   correct the list with the user, and commit again.

7. **Write `search-space.md`** in the experiment folder from
   `.finetune-kit/templates/search-space.md`: the round, each open setting and why (★ reason or
   the user's reason), the allowed losses, and what stays fixed. On a new round, keep the earlier
   rounds' history in the file and add the new round below it.

## Finish

If `experiments/<exp>/baseline.md` does not exist yet, the next command is `{{CMD}}baseline`.
If it does (a new round), the next command is `{{CMD}}research`. Tell the user which.
