# finetune-kit: status

Step 12 of 14, usable any time. Show where the experiment stands and how the experiments are
going, in plain words. This step only reads; it never starts, stops or changes anything.

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

4. **Results so far.** If the experiments have started (step 11), run `finetune-kit compare <exp>`
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

5. If research is running in another chat or window, say so plainly: do not start the research
   loop here as well; the next step below is for when that one has stopped.

## Finish

{{FINISH}}
