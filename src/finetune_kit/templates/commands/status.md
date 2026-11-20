# finetune-kit: status

Step 12 of 14, usable any time. Show where the experiment stands and how the experiments are
going, in plain words. This step only reads; it never starts, stops or changes anything.

## Before you start

- `.finetune-kit/memory/research-rules.md` must exist. If it does not, finetune-kit is not
  installed in this project: tell the user to run `finetune-kit init --ai <their agent> --here`,
  and stop.
- Run `finetune-kit version`. If the command is not found, tell the user to install the CLI with
  `uv tool install` (see the finetune-kit README), and stop.

## How to work in this step

{{HOW_TO_WORK}}

## Steps

1. Run `finetune-kit steps <exp>` (no argument if the user named no experiment; it picks the
   highest-numbered one). If `experiment` is `null`, there is no experiment yet: say so, and go
   to Finish.

2. **Where they are.** Say which steps are done (`status` `done`), which one is next
   (`next.stage`, with its `number` and `title` from `steps`) and why (`next.reason`, in plain
   words).

3. **Running now.** Run `finetune-kit status <exp>`. Say whether something is training now
   (`running`: which kind of run, since when) and how many runs wait in the queue (`queue`).

4. **Results so far.** If the experiments have started (Step 11 of 14 — Run the
   experiments), run `finetune-kit compare <exp>`
   and summarize in a few plain lines:
   - the starting point: the `noise` run's mean score, and the keep bar (`noise.keep_bar`) a
     change must beat;
   - experiments done: how many research runs finished, and how many were kept (`verdict`
     `keep` or `keep-simpler`) versus discarded or crashed;
   - the best score so far (`best` run's mean ± std) and how much better it is than the starting
     point;
   - budget left (`budget_left`: runs and hours) and how close research is to stopping on its
     own (`streak` of non-keeps in a row, out of 10; `stop`).

   If `experiments/<exp>/log.md` exists, you may read it to say in one line what the latest run
   tried. Before the experiments start, skip this and say there are no results yet.

5. **Is research running?** If `next.stage` is `start-experiment` and the experiments have
   started (or the starting-point measurement is running), look at the `finetune-kit status`
   reply from step 3 above: if a run of this experiment is `running` or in the `queue`, research
   is going on (in another chat or window, or in the background). Say so plainly. Never offer
   to start a second research loop while one runs. If nothing of this experiment is running or
   queued, research has paused: `{{CMD}}start-experiment` resumes it where it stopped.

## Finish

If step 5 above found research running, do not offer any step: tell the user research carries on
by itself, and that they can check again any time with `Command: {{CMD}}status`. Then stop.
Otherwise:

{{FINISH}}
