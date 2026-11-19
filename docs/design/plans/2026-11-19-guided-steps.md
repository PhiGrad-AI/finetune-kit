# finetune-kit guided steps Implementation Plan

**Goal:** Replace the eight engineer-named stages with fifteen plain-language steps, one decision each, with a summary at the start and the next-step hand-off after every step.

**Architecture:** An experiment is created at `choose-task` with only its task and filled one answer at a time through `finetune-kit set <exp> <what> …`; answers live in `experiment.yaml` until the data is checked (then locked) and, for choices that change between research rounds, in `state.json`. `finetune-kit next` follows the new step order; `finetune-kit steps` lists all steps with their status. One prompt per step replaces the eight stage prompts.

**Tech Stack:** existing (Python 3.12, typer, pydantic, pytest).

**Spec:** `docs/design/specs/2026-11-19-guided-steps-design.md`

## Global Constraints

- Step names, in order: `start`, `choose-task`, `choose-data`, `choose-model`, `choose-goal`, `check-data`, `compare-models`, `choose-loss`, `choose-hyperparameters`, `choose-learning-rate`, `choose-experiments`, `start-experiment`, `status`, `report`, `finish`. `compare-models` only applies when 2–3 models were chosen; `status` is available any time research has started; `start` is the introduction.
- Commands per agent: Claude Code `/finetune-kit-<step>`, Codex `$finetune-kit-<step>`, Cursor `/finetune-kit.<step>`.
- Every CLI command prints exactly one JSON object; errors `{"error": "<message>"}`, exit 2, one line.
- Experiments created by earlier versions (all of task, data, model, objective set) keep loading and map onto the new order.
- The one-shot `finetune-kit new NAME --task … --model … --train/--source … --metric …` keeps working.
- Commits and PR carry no AI attribution (no Co-Authored-By trailer, no "Generated with" line).
- Unchanged: keep rule, test vault, checksums from the noise run on, `--losses` semantics (`ce` always allowed; a second loss makes `loss` tunable).

## Review Focus

1. **Changing an answer after the data is checked** (`set data/model/goal` after splits are frozen) → one-line error naming why (`data is checked and locked; start a new experiment to change it`). Test in Task 1.
2. **Running a later CLI step on an incomplete experiment** (`check-data` with no model or goal) → one-line error naming the missing step (`choose the model first (step: choose-model)`), nothing written. Test in Task 1.
3. **Round 2** (report → choose-loss → choose-hyperparameters) → `next` walks choose-loss → choose-hyperparameters → start-experiment, and changing losses in round 2 does not trip the experiment.yaml tamper check. Test in Task 2/3.
4. **An experiment from the old flow** (created with one-shot `new`, noise measured) → `next` and `steps` work and show earlier steps done. Test in Task 3.
5. **Reinstalling** over an old install → the eight old prompt files that `init` wrote are removed, new ones written, user files untouched. Test in Task 4.

---

### Task 1: Step-by-step experiment: `new --task` and `set data|model|goal`

**Files:** `src/finetune_kit/experiment.py`, `src/finetune_kit/commands.py`, `src/finetune_kit/cli.py`; tests in `tests/test_cli_commands.py`, `tests/test_experiment.py`.

**Interfaces:**
- `ExperimentConfig.model: str | None = None`, `data: DataPaths | None = None`, `objective: MetricRef | None = None` (draft until set). `DataPaths` validation unchanged when present.
- `ex.missing_setup(cfg) -> str | None`: first unset of data → `"choose-data"`, model → `"choose-model"`, objective → `"choose-goal"`.
- `ex.require_setup(cfg) -> None`: raises `ValueError(f"{what} first (step: {step})")` with `what` = "choose the data" / "choose the model" / "choose the goal". Called by `check-data`, `catalog`, `bakeoff`, `lr-scan`, `run`, `promote`, `misses`, `compare` (any command that needs them).
- `commands.setup_experiment` keeps its signature; when only `task` is given it writes a draft; otherwise behaves as today.
- CLI `new NAME --task T` (other options optional as today; given together = one-shot).
- CLI sub-app `set`: `set data <exp> (--train P [--val P --test P] | --source hf:… [--revision R]) [--columns JSON]`; `set model <exp> MODEL [--compare a,b[,c]]` (stores `model` and, with `--compare`, `bakeoff` = [MODEL, …] with today's validation); `set goal <exp> --metric M [--metric-params JSON] [--guardrail G] [--guardrail-params JSON] [--positive-class C] [--custom-metric PATH:FUNC --direction max|min]`. Each validates exactly as `new` does today (reuse the same helpers, no copies), rewrites `experiment.yaml` atomically under the state lock, and prints the updated part. Each refuses once `splits/train.jsonl` exists: `"<what> is checked and locked; start a new experiment to change it"`. `set goal` and `set model` that depend on data (positive class, label checks) require data first (`choose the data first (step: choose-data)`).
- [ ] Failing tests: draft `new --task`; each `set` happy path and its validation error; set after check refused; `check-data` on a draft refused with the step name and nothing written; one-shot `new` unchanged; old experiment.yaml loads.
- [ ] Implement; `uv run pytest -q tests/test_cli_commands.py tests/test_experiment.py`; full suite; commit `feat: build an experiment one answer at a time`.

### Task 2: Choices that can change per round: `set losses|learning-rate|experiments`

**Files:** `experiment.py` (State), `cli.py`, `commands.py`; tests `tests/test_cli_commands.py`.

**Interfaces:**
- `State.chosen_losses: list[str] | None = None`, `State.lr_choice: Literal["default", "scan"] | None = None`, `State.budget_chosen: bool = False`.
- `set losses <exp> ce,focal`: validated against the task's losses; `ce` always added; stored in `state.chosen_losses` (state.json, so round 2 does not touch experiment.yaml); allowed until the round's search space is committed, again after each report (new round). Prints `{"losses": [...]}`.
- `space commit <exp> --tunable …`: `--losses` becomes optional when `state.chosen_losses` is set (uses it); if neither → today's `--losses is required…` error. Still adds `loss` when a second loss is allowed.
- `set learning-rate <exp> default|scan`: stored in `state.lr_choice`; refused after the noise run (`the learning rate is fixed once the experiment has started`). `scan` does not run anything itself; the step prompt runs `lr-scan` after setting it.
- `set experiments <exp> [--max-runs N] [--max-hours H] [--seeds S] [--subset N]`: writes budget/seeds/research_subset into experiment.yaml (validated as `new` does: seeds ≥ 2 etc.), sets `state.budget_chosen = True`; refused after the noise run.
- [ ] Failing tests for each, including round-2 `set losses` after a noise run leaving `verify_checksums` passing, and `space commit` without `--losses` using chosen losses.
- [ ] Implement; tests; full suite; commit `feat: choose losses, learning rate and experiment budget as separate answers`.

### Task 3: `next` and `steps` follow the new order

**Files:** `commands.py`, `cli.py`; tests `tests/test_cli_commands.py`.

**Interfaces:**
- `commands.STEPS: list[dict]` — each `{"step", "title", "what"}` (plain one-line description a non-engineer understands) in the Global Constraints order.
- `next_stage` returns new step names. Order (first unmet wins): no experiment → `choose-task` ("no experiment yet"); `missing_setup(cfg)` → that step; splits missing → `check-data`; `cfg.bakeoff and not state.model_locked` → `compare-models`; no search space for the current round and `state.chosen_losses is None` → `choose-loss`; no search space for the current round → `choose-hyperparameters`; `state.sigma_obj is None and state.lr_choice is None` → `choose-learning-rate`; `state.sigma_obj is None and state.lr_choice == "scan" and state.lr_scan_best is None` → `choose-learning-rate` ("learning-rate scan not finished"); `state.sigma_obj is None and not state.budget_chosen` → `choose-experiments`; `state.sigma_obj is None` → `start-experiment`; research not finished → `start-experiment` ("research is still running", alternative `status`); report stale → `report` (alternative `choose-loss`, "try other losses or settings in another round"); not promoted → `finish` (same alternative); promoted → `choose-task` ("this experiment is complete; start a new one"). Old experiments (noise measured) never fall into the lr/budget rules because those are gated on `sigma_obj is None`.
- New rounds: `State.round_open: bool = False`. `set losses` after the noise run sets `round_open = True` (a new round has started); `space commit` sets it back to `False`. "No search space for the current round" means `search-space.yaml` missing or `state.round_open`. While `round_open`, `chosen_losses` is the new round's choice, so the `choose-loss` rule is skipped and `choose-hyperparameters` comes next.
- CLI `finetune-kit steps [EXP]` → `{"experiment": …, "steps": [{"step", "title", "what", "status": "done"|"next"|"to do"|"optional"|"skipped"}], "next": <same as next>}`; `compare-models` is "skipped" without a comparison, "optional" before the model step; `status` is "optional".
- [ ] Failing tests: full walk on a fresh experiment through every rule; round 2 walk; old one-shot experiment after noise; `steps` statuses at three points.
- [ ] Implement; tests; full suite; commit `feat: next and steps use the guided step order`.

### Task 4: One prompt per step

**Files:** delete `src/finetune_kit/templates/commands/{setup,check,bakeoff,hparams,baseline,research,analyze,promote}.md`; create one template per step (15); update `_goto.md`/`_finish.md`, `installer.py` (`STAGES` → steps, `DESCRIPTIONS`, removal of old prompt files on `init`), `templates/memory/research-rules.md`, `guard.py` if it lists stage names; tests `tests/test_installer.py`.

- Content moves, not rewritten from scratch: choose-task/choose-data/choose-model/choose-goal from setup.md (one question each, `new --task` then `set …`); check-data from check.md; compare-models from bakeoff.md; choose-loss and choose-hyperparameters from hparams.md (the loss question and the settings question respectively; `set losses`, then `space commit --tunable`); choose-learning-rate from baseline.md's LR-scan part; choose-experiments from setup.md's seeds/budget/subset questions; start-experiment from baseline.md's noise part + research.md; status (new: `finetune-kit steps`, `status`, `compare`, in plain words); report from analyze.md; finish from promote.md; start (new: run `finetune-kit steps`, show the summary as a numbered list with each step's one-line `what`, then offer the next step).
- Every step except start and status begins with the shared order check; every step ends with the shared hand-off, which also shows "Step N of 14".
- `init` removes the old eight prompt files/folders it wrote previously (only paths it owns) and writes the new ones.
- [ ] Failing tests: every step renders for every agent with no `{{`; each prompt asks only its own question (assert key question text per step); start lists all steps; hand-off shows "Step"; old files removed on re-init and a user file next to them kept.
- [ ] Implement; tests; full suite; commit `feat: one plain-language prompt per step`.

### Task 5: README, rules and end-to-end

**Files:** `README.md`, `tests/test_e2e.py`.
- README: the step table (command, what you decide), "start with `$finetune-kit-start`", the `set` commands, `steps`, `next`.
- e2e: tiny-bert seq-cls through the CLI the way the prompts drive it: `new --task`, `set data`, `set model`, `set goal`, `check-data`, `set losses`, `space commit --tunable`, `set learning-rate default`, `set experiments --seeds 2 …`, noise run ok; assert `next` at each point.
- [ ] Implement; full suite; commit `docs: guided steps in the README; end-to-end through the new steps`.
