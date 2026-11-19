Hand off to the next stage. Do this every time, in this order:

1. Run `finetune-kit next <exp>`. It prints `stage`, `reason` and `alternatives` (a list of
   `{stage, reason}`), computed from the experiment's files. Trust it, not your own guess.
2. Tell the user, in one line, what this stage did. Then show the next stage like this, and list
   each alternative the same way:
   `Next: <stage> - <reason>. Command: {{CMD}}<stage>`
3. Ask: "Shall I start <stage> now?" and wait for the answer.
4. If the user says yes, open `{{STAGE_FILE}}` (with `<stage>` replaced by that stage's name)
   and follow it from the top, including its "Before you start" checks. If the user picks one
   of the alternatives instead, open that stage's file instead.
5. If the user says no, stop. Tell them they can run the command whenever they like.
6. Never start a stage the user has not agreed to, and never skip one.
