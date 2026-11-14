# Baseline: <NNN-name>

## Learning-rate scan

<Skipped (learning rate is not tunable), or:>

| Learning rate | Objective (1 seed, 20% of training) |
|---|---|
| <lr> | <value, or crash> |

Chosen default learning rate: <lr>. <One line: these are single short runs, so this is only a
starting point.>

## Noise floor

Run <run id>: the default recipe on <N> seeds.

| Metric | Mean | σ (spread across seeds) |
|---|---|---|
| <objective> | <mean> | <σ> |
| <guardrail, if any> | <mean> | <σ_guard> |

**Keep bar:** a change must improve <objective> by more than <keep_bar> to be kept.
<If there is a guardrail: it may not drop by more than <σ_guard · sqrt(2/N)>.>

Time per run: about <minutes> minutes. Runs longer than 3× the baseline training time per seed
are stopped. Budget left: <runs> runs, <hours> hours (about <estimate> runs at this speed).

## What this means

<Plain words: how noisy the metric is, and what size of gain research can hope to prove.>

## Next

Research (runs on its own until the stop rule).
