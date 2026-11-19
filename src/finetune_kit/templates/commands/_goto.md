This is the `{{STAGE}}` stage. Check the stage order first. The stages, in order:
{{STAGE_ORDER}}.

1. Run `finetune-kit next <exp>` (if there is no experiment folder yet, run `finetune-kit next`
   with no argument). It prints `stage` and `reason`, computed from the experiment's files.
   Trust it, not your own guess.
2. {{CONTINUE_WHEN}}
3. Otherwise, if `stage` is an earlier stage, do not start this stage: that stage is not done
   yet. Earlier stages: {{EARLIER}}.
   Show it: `Next: <stage> - <reason>. Command: {{CMD}}<stage>`.
   Ask "Shall I start <stage> now?" and wait. If the user says yes, open `{{STAGE_FILE}}`
   (with `<stage>` replaced by that stage's name) and follow it from the top. If no, stop.
   Never start a stage the user has not agreed to.

Only `finetune-kit next` decides whether this stage may start. A missing write-up from an earlier
stage (`experiment.md`, `data-report.md`, `bakeoff.md`, `search-space.md`, `baseline.md`,
`log.md`, `report.md`) never holds it back. If this stage needs something from one that is
missing, tell the user it is missing, get what you need from finetune-kit instead
(`finetune-kit compare`, `finetune-kit catalog`, `data-report.json`, `search-space.yaml`,
`experiment.yaml`; reading them is fine), and go on.
