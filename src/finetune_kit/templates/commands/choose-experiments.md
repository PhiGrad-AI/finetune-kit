# finetune-kit: choose-experiments

Step 10 of 14. Decide how much research may try: how many experiments and hours at most, how
many times each experiment is repeated (seeds), and, for big datasets, whether research uses a
smaller part of the data to go faster. Record it.

## Before you start

{{ORDER_CHECK}}

If `set experiments` says it is refused because the experiments have started, the budget is
fixed for this experiment: say so in plain words and use the hand-off in the Finish section.

## How to work in this step

{{HOW_TO_WORK}}

## Steps

**First, are the sample size and repeats already fixed?** Run `finetune-kit compare <exp>` and
`finetune-kit status <exp>`. If either shows an `lr-scan` or `bakeoff` run (finished, queued or
running), the model comparison or the learning-speed test already trained with the current seeds
and research subset (`seeds` and `research_subset` in `experiments/<exp>/experiment.yaml`;
reading is fine), so they are fixed. Show them to the user as already fixed, skip step 1 below
and step 3 below, and leave `--seeds` and `--subset` out in step 4 below: ask only the number of
experiments and hours. If they were answered earlier but nothing has trained yet (the *Seeds per
run* and *Research subset* rows of `experiment.md` are filled), show them and ask only whether to
keep them.

1. **Seeds per experiment.** Explain: every experiment trains the same recipe several times with
   different random starting points (seeds) and averages the results, because one training run
   alone can be lucky. More seeds give more trustworthy comparisons but take longer. Default
   **3**; at least 2. Ask how many seeds they want, and wait.

2. **Budget.** Ask: "How many experiments may research run at most, and for how many hours?"
   (defaults: 40 experiments, 24 hours). Explain that research stops earlier on its own when ten
   experiments in a row bring no real improvement. Wait for the answer.

3. **Research subset.** Read the training row count from `experiments/<exp>/data-report.json`
   (`n_rows.train`; reading is fine). Only if it is more than 5,000 rows: offer to run the
   research on a random subset (for example 3,000 rows) to make each experiment faster, and explain that the
   final model is still trained on all the data at the end. Skip this question otherwise.

4. **Record it.** Show the answers as a short list, then run
   `finetune-kit set experiments <exp> --seeds <N> --max-runs <runs> --max-hours <hours> [--subset <rows>]`.
   On an error (for example fewer than 2 seeds), explain it, ask the one question that fixes it,
   and run it again.

5. **Update `experiment.md`:** fill in the *Seeds per run*, *Budget* and *Research subset* rows
   with their reasons.

## Finish

{{FINISH}}
