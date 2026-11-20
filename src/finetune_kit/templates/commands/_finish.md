Hand off to the next step. Do this every time, in this order:

1. Run `finetune-kit steps <exp>` (with no argument if there is no experiment yet). Its `next`
   field is exactly what `finetune-kit next <exp>` prints: `stage`, `reason` and `alternatives`
   (a list of `{stage, reason}`), computed from the experiment's files. Trust it, not your own
   guess. For the step named in `next.stage` (and each alternative), find its `number` and
   `title` in `steps`. If you already ran `finetune-kit steps` in this step and nothing has
   changed since, reuse that reply.
2. Tell the user, in one line, what this step did. Then show the next step like this, and list
   each alternative the same way:
   `Next: Step <number> of 14 — <title> (<reason>). Command: {{CMD}}<step>`
   where `<step>` is the step's name (`next.stage`) and `<reason>` is said in plain words.
3. Ask: "Shall I start <title> now?" and wait for the answer.
4. If the user says yes, open `{{STEP_FILE}}`
   (with `<step>` replaced by that step's name) and follow it from the top, including its "Before you start" checks. If the user picks one
   of the alternatives instead, open that step's file instead.
5. If the user says no, stop. Tell them they can run the command whenever they like.
6. Never start a step the user has not agreed to, and never skip one.
