Go to the stage that comes first. Run `finetune-kit next <exp>` (if there is no experiment
folder yet, run `finetune-kit next` with no argument) and show the stage it names with its
reason and command: `Next: <stage> - <reason>. Command: {{CMD}}<stage>`. Then ask
"Shall I start <stage> now?" and wait. If the user says yes, open `{{STAGE_FILE}}` (with
`<stage>` replaced by that stage's name) and follow it from the top. If no, stop. Never start
a stage the user has not agreed to.
