# finetune-kit: guided hand-off between stages

Short design, follow-up to `2026-11-14-finetune-kit-design.md`.

## Why

In a live Codex run the agent never asked which hyperparameters to tune. Each stage prompt
ends with "Next command: …" and stops, so a stage only runs if the user types its command;
`hparams` (where the tunable settings are chosen) was skipped. The user wants every stage to
end by offering the next one, continuing in the same conversation on a yes, and always showing
the command so they can run it themselves later.

## Goals

- Every stage ends with: what was done, the next stage and why, its command (e.g.
  `$finetune-kit-hparams`), and the question "Shall I start it now?".
- On a yes, the agent continues with the next stage's own prompt, from its first line, in the
  same conversation. On a no, it stops; the user has the command.
- Which stage comes next is decided by the CLI from the experiment's state, not guessed by the
  agent, so no stage (in particular `hparams`) can be skipped.

## Non-goals

- Running several stages without asking. The user confirms every hand-off.
- Changing what any stage does internally.

## Approaches

- **A. Prompts only:** each stage asks to continue and opens the next prompt.
- **B. A plus a `finetune-kit next` command** that computes the next stage.
- **Chosen: both (B includes A)**, plus the command is always shown — the user's words: "we
  should do both and also user should know the next command".

## Design

**`finetune-kit next [<exp>]`** prints one JSON object:
`{"experiment": "001-x", "stage": "hparams", "reason": "no search space committed yet",
"alternatives": [{"stage": "...", "reason": "..."}]}`. With no experiment (none exists, or the
last one is promoted) the stage is `setup`. The order, first unmet wins:

1. no frozen splits → `check`
2. a bake-off was requested and none has been chosen → `bakeoff`
3. no committed search space → `hparams`
4. no noise floor measured → `baseline`
5. research not finished (no `## Summary` in `log.md`, budget left, stop rule not hit) →
   `research`
6. no `report.md` → `analyze` (alternative: `hparams` to open more settings, new round)
7. not promoted → `promote` (alternative: `hparams`)
8. promoted → `setup` for a new experiment

**Bake-off is recorded in the config.** `new` gains `--bakeoff <model,model,...>` (stored as
`bakeoff: [...]` in `experiment.yaml`; absent for older experiments, meaning no bake-off), so
the CLI knows a bake-off was requested instead of reading prose in `experiment.md`.

**Prompt hand-off.** Every stage's *Finish* section is replaced by one shared block: run
`finetune-kit next <exp>`; tell the user the next stage, the reason, and the command
`{{CMD}}<stage>` (plus any alternatives with their commands); ask "Shall I start it now?". On
yes, read the next stage's prompt file and follow it from the top; the installer fills in the
per-agent path (`.claude/skills/finetune-kit-<stage>/SKILL.md`,
`.agents/skills/finetune-kit-<stage>/SKILL.md`, `.cursor/commands/finetune-kit.<stage>.md`).
Each stage's "Before you start" checks stay, so a stage entered by hand still refuses to run
out of order. `research-rules.md` gains the rule: never skip a stage; always confirm before
moving on.

## Left to the builder

- Exact wording of the shared Finish block (one place, rendered into all eight prompts).
- How `next` detects "stop rule hit" (reuse whatever `compare` already computes).
