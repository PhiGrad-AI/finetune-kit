# finetune-kit stage hand-off Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every stage ends by naming the next stage (decided by a new `finetune-kit next` command), showing its command, and asking to continue in the same conversation.

**Architecture:** `commands.next_stage(root, exp)` computes the next stage from experiment files and state; CLI `next` prints it. `new --bakeoff` records a requested bake-off in `experiment.yaml`. One shared Finish template is rendered into all eight stage prompts by the installer with per-agent command prefix and prompt-file path.

**Tech Stack:** existing (Python 3.12, typer, pydantic, pytest).

**Spec:** `docs/superpowers/specs/2026-11-19-stage-handoff-design.md`

## Global Constraints

- Every CLI command prints exactly one JSON object; errors `{"error": "<message>"}`, exit 2.
- Stage names: `setup`, `check`, `bakeoff`, `hparams`, `baseline`, `research`, `analyze`, `promote`.
- Commands per agent: Claude Code `/finetune-kit-<stage>`, Codex `$finetune-kit-<stage>`, Cursor `/finetune-kit.<stage>`.
- Prompt-file paths per agent: Claude `.claude/skills/finetune-kit-<stage>/SKILL.md`, Codex `.agents/skills/finetune-kit-<stage>/SKILL.md`, Cursor `.cursor/commands/finetune-kit.<stage>.md`.
- The agent never moves to another stage without the user's yes; the next command is always shown.
- Experiments written by earlier versions (no `bakeoff` key) load and mean "no bake-off".

## Review Focus

1. **New research round** (user reran `hparams`, `state.round` advanced) → `next` says `research`, not `analyze`/`promote`, even though `log.md` already has an earlier summary and `report.md` exists. Test in Task 1.
2. **Bake-off requested but not yet chosen** → `next` says `bakeoff` before `hparams`; after `bakeoff choose` (or noise, which locks the model) it moves on. Test in Task 1.
3. **Promote queued but not finished** (`state.promote_pending`) → `next` says `promote` with reason "promote is running"; after `promoted` → `setup`. Test in Task 1.
4. **No experiments folder / empty project** → `next` with no argument returns `setup`, not an error. Test in Task 1.
5. **Agent says no to continuing** → the prompt tells it to stop after showing the command, not to start anything. Assert the wording in Task 2's installer test.

---

### Task 1: `finetune-kit next` and `new --bakeoff`

**Files:**
- Modify: `src/finetune_kit/experiment.py` (`ExperimentConfig.bakeoff: list[str] | None = None`)
- Modify: `src/finetune_kit/commands.py` (`setup_experiment(..., bakeoff: list[str] | None = None)`, `next_stage`)
- Modify: `src/finetune_kit/cli.py` (`new --bakeoff`, `next` command)
- Test: `tests/test_cli_commands.py`

**Interfaces:**
- Produces: `commands.next_stage(root: Path, exp_dir: Path | None) -> dict` returning `{"experiment": str | None, "stage": str, "reason": str, "alternatives": [{"stage": str, "reason": str}]}`.
- CLI: `finetune-kit next [EXP]` (EXP optional: highest-numbered experiment; none → `{"experiment": null, "stage": "setup", "reason": "no experiment yet", "alternatives": []}`).
- `new --bakeoff a,b[,c]`: 2–3 model profile keys, each valid for the task, first equal to `--model` (error otherwise: `"--bakeoff needs 2-3 models for <task>, the first being --model"`); stored as `bakeoff: [..]`.

Order (first unmet wins), with exact reasons:
1. `splits/train.jsonl` missing → `check`, "data not checked and split yet"
2. `cfg.bakeoff` and not `state.model_locked` → `bakeoff`, "a bake-off was requested and no model is chosen yet"
3. `search-space.yaml` missing → `hparams`, "no search space committed yet"
4. `state.sigma_obj is None` (no noise floor) → `baseline`, "noise floor not measured yet"
5. research not finished for `state.round` → `research`, "research round N is not finished". Finished means `log.md` contains a line starting `## Summary (round N)`; for `N <= 1` a bare `## Summary` line also counts.
6. `report.md` missing or older (mtime) than `log.md` → `analyze`, "results not analyzed yet"; alternative `hparams`, "open more settings for another research round"
7. `state.promote_pending` → `promote`, "promote is running"
8. not `state.promoted` → `promote`, "final model not trained and scored yet"; alternative `hparams` as above
9. promoted → `setup`, "this experiment is complete; start a new one"

- [ ] **Step 1: Failing tests** in `tests/test_cli_commands.py` building experiments on disk with the existing helpers (write files / set state with `ex.save_state` rather than training): `test_next_no_experiment_is_setup`, `test_next_order_check_then_hparams`, `test_next_bakeoff_before_hparams_until_locked`, `test_next_baseline_then_research`, `test_next_research_done_then_analyze_with_alternative`, `test_next_new_round_returns_research` (round 2, old `## Summary`, existing report.md → `research`), `test_next_promote_pending_then_done`, `test_new_bakeoff_validated_and_saved`, `test_old_yaml_without_bakeoff_loads`.
- [ ] **Step 2:** Run them — expect failures.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** `uv run pytest -q tests/test_cli_commands.py tests/test_experiment.py` — PASS.
- [ ] **Step 5:** Commit `feat: finetune-kit next says which stage comes next`.

### Task 2: Shared hand-off in every stage prompt

**Files:**
- Create: `src/finetune_kit/templates/commands/_finish.md`
- Modify: all eight `src/finetune_kit/templates/commands/<stage>.md` (replace each `## Finish` body's "next command" lines with `{{FINISH}}`; keep stage-specific closing content such as research's `## Summary` instructions)
- Modify: `src/finetune_kit/templates/commands/research.md` (summary heading becomes `## Summary (round N)`), `src/finetune_kit/templates/commands/setup.md` (pass `--bakeoff` when chosen), `src/finetune_kit/templates/memory/research-rules.md` (rule: never skip a stage; confirm before moving on)
- Modify: `src/finetune_kit/installer.py` (`render` fills `{{FINISH}}`, then `{{CMD}}` and `{{STAGE_FILE}}`)
- Modify: `README.md` (stages section: the hand-off and `finetune-kit next`)
- Test: `tests/test_installer.py`

**Interfaces:**
- Consumes: CLI `finetune-kit next <exp>` (Task 1).
- `{{STAGE_FILE}}` renders to the agent's prompt-file path with a literal `<stage>` placeholder, e.g. `.agents/skills/finetune-kit-<stage>/SKILL.md`.

Shared Finish text (exact meaning; wording may be tightened):
1. Run `finetune-kit next <exp>`.
2. Tell the user in one line what this stage did; then "Next: <stage> — <reason>. Command: `{{CMD}}<stage>`". List each alternative the same way.
3. Ask: "Shall I start <stage> now?" and wait.
4. Yes → open `{{STAGE_FILE}}` (with `<stage>` replaced) and follow it from the top, including its "Before you start" checks. If the user picks an alternative, open that one instead.
5. No → stop. They can run the command whenever they like.
6. Never start a stage the user has not agreed to; never skip one.

- [ ] **Step 1: Failing tests** in `tests/test_installer.py`: for each agent and each stage, the rendered prompt contains `finetune-kit next`, `Shall I start`, the agent's own command prefix, and its own `{{STAGE_FILE}}` path; contains no unreplaced `{{`; research prompt contains `## Summary (round`; setup prompt mentions `--bakeoff`; research-rules mentions never skipping a stage.
- [ ] **Step 2:** Run — expect failures.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** `uv run pytest -q tests/test_installer.py`, then full suite `uv run pytest -q`.
- [ ] **Step 5:** Commit `feat: every stage offers the next one and shows its command`.
