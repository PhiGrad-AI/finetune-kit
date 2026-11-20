This is the `{{STAGE}}` step. Check the step order first. The steps, in order:
{{STAGE_ORDER}}.

1. Run `finetune-kit next <exp>` (if there is no experiment folder yet, run `finetune-kit next`
   with no argument). It prints `stage` and `reason`, computed from the experiment's files.
   Trust it, not your own guess.
2. {{CONTINUE_WHEN}}
3. Otherwise, if `stage` is an earlier step, do not start this step: that step is not done
   yet. Earlier steps:
   {{EARLIER}}.
   Run `finetune-kit steps <exp>` and find that step in `steps` for its `number` and `title`.
   Show it: `Next: Step <number> of 14 — <title> (<reason>). Command: {{CMD}}<step>`.
   Ask "Shall I start <title> now?" and wait. If the user says yes, open
   `{{STEP_FILE}}` (with `<step>` replaced by that step's name) and follow it
   from the top. If no, stop.
   Never start a step the user has not agreed to.

Only `finetune-kit next` decides whether this step may start. A missing write-up from an earlier
step (`experiment.md`, `data-report.md`, `bakeoff.md`, `search-space.md`, `baseline.md`,
`log.md`, `report.md`) never holds it back. If this step needs something from one that is
missing, tell the user it is missing, get what you need from finetune-kit instead
(`finetune-kit compare`, `finetune-kit catalog`, `data-report.json`, `search-space.yaml`,
`experiment.yaml`; reading them is fine), and go on.
