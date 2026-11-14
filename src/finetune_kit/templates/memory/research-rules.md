# finetune-kit research rules

These rules bind the coding agent in every finetune-kit stage. The `finetune-kit` CLI enforces
the numbers (search space, hard limits, seeds, budget, keep verdict, test isolation); the agent
follows the rest.

You (the user) may edit this file, but only before an experiment's data check
(`finetune-kit check-data`): from then on its checksum is recorded, and `finetune-kit run`
refuses to start if the file changes.

## The six research rules

1. **One idea per run.** Change one idea at a time, so each result teaches something. Give every
   run a one-line description (`--desc`) and write a one-line hypothesis in `log.md` before the
   result is known.
2. **Only the search space moves.** Change only settings listed in the experiment's
   `search-space.yaml`. Everything else stays at the profile default. To open more settings,
   the user reruns the hparams stage, which starts a new round.
3. **Keep only what beats the noise.** A run is kept only if it beats the best run by more than
   the noise bar and does not hurt the guardrail metric beyond its own noise (exact rule below).
   Otherwise it is discarded and the next run starts again from the last kept config. An equal
   result with a simpler config is kept ("keep-simpler").
4. **Crashes and timeouts are discards.** Note the error, and never retry the same config
   unchanged.
5. **Know when to stop.** Stop after 10 non-keeps in a row, or when the budget (runs or hours)
   is spent. Then write the summary. Never ask the human whether to continue.
6. **Hands off the evidence.** Never read test data. Never change splits, metrics or these rules.

## The keep rule (computed by `finetune-kit`, never by the agent)

Every run trains N seeds (N = `seeds` in `experiment.yaml`) and is compared by its mean.
σ is the standard deviation of the objective across the seeds of the noise-floor run
(the default config); σ_guard is the same for the guardrail metric.

```
noise = σ · sqrt(2/N)
Δ     = new mean − best mean        (flipped for lower-is-better metrics)

keep          iff  Δ > max(2·noise, min_delta)
                   and guardrail drop ≤ σ_guard · sqrt(2/N)
keep-simpler  iff  |Δ| ≤ 2·noise and the config has fewer non-default keys than the kept one
discard       otherwise (crash and timeout are discards too)
```

`min_delta` is 0.002 for metrics in [0, 1] unless `experiment.yaml` overrides it.
`finetune-kit compare` shows the resulting bar as `keep_bar`. The verdict printed by
`finetune-kit wait` / `finetune-kit compare` is final: never override or reinterpret it.

## The stop rule

Research stops when `finetune-kit compare` reports `"stop": true`, which happens after
**10 non-keeps in a row** or when the **budget is spent** (runs or hours). A run that exceeds
3× the noise-floor run's training time per seed is stopped and counts as a timeout.

## How the agent works

- Ask the user one question at a time, in plain language. Explain any ML term in a few words.
- Act only through `finetune-kit …` commands. Each prints one JSON object; read it before acting.
- On `{"error": "..."}`, explain the problem to the user in plain words, then fix it or stop.
- Long runs detach. Poll them with `finetune-kit wait <run_id> --exp <exp> --timeout 540`,
  again and again, until the status is `done`, `crash` or `timeout`.
- Never edit files that finetune-kit writes: anything under `.finetune-kit/memory/`,
  `experiments/*/splits/`, `experiments/*/runs/`, `experiments/*/final/`, and
  `experiments/*/{search-space.yaml, state.json, results.tsv, experiment.yaml, data-report.json}`.
- Never open anything under `~/.finetune-kit/` (it holds the test split), `~/.cache/huggingface/`
  (it may hold downloaded datasets) or `experiments/*/splits/`. Do not open the user's dataset
  files either; `finetune-kit check-data` inspects them.
- The agent writes only its own notes: `experiment.md`, `data-report.md`, `bakeoff.md`,
  `search-space.md`, `baseline.md`, `log.md` and `report.md` in the experiment folder.
