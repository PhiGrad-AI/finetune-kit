# finetune-kit: agent-driven fine-tuning research for BERT-family encoders

Date: 2026-11-14 · Status: draft for review · Name: finetune-kit; command `finetune-kit`, import package `finetune_kit`

## 1. Intent

An earlier project showed that careful, Karpathy-style
autoresearch on an encoder model pays off: 116 runs took a classifier from test `pauc_hi` 0.71
(untuned) to 0.93 (LLRD 0.75, lr 4e-5). It also showed the traps: one run varies by about
±0.05, so single runs prove nothing; model quirks (DeBERTa fp16 on MPS) waste hours; and the
search only works when the agent is boxed in by clear rules.

`finetune-kit` packages that know-how so anyone can do the same on their own dataset, using the
coding agent they already have (Claude Code, Codex or Cursor) as the researcher. The person
says what they care about (task, data, model, metric, which knobs may move); the agent picks
values, runs experiments, reads results and decides what to try next; the tool enforces the
rules and keeps an honest record. The output is a final recipe backed by multi-seed evidence.

**Why terminal + agent, not a web UI:** the agent is the interface. No server, no UI to
build, and it works the same in all three agents. A UI can be added later on top of the
same CLI.

## 2. Goals, non-goals, anti-goals

**Goals (v1)**
- Two tasks: sequence classification (binary and multi-class) and token classification (NER).
- A spec-kit-style flow of eight slash commands, each leaving a reviewable file behind.
- Installs into any project for Claude Code, Codex or Cursor with one command.
- A general trainer covering the hyperparameters and losses used in the PII sweep, plus
  a few proven additions.
- User picks the metric (from a list or a custom function) and which hyperparameters are
  tunable. The agent picks values.
- Noise-aware decisions: every comparison uses multiple seeds and a σ-based keep rule.
- Runs locally on whatever the machine has: CUDA, Apple MPS or CPU.

**Non-goals (v1)**
- Web UI, remote GPU job queue, cloud launching.
- Multi-label classification, seq2seq, decoder LLM fine-tuning.
- Automatic data windowing/chunking of long documents (user prepares windows; the tool reports
  what is too long).
- A plain Optuna/random sweep mode (possible later as `finetune-kit sweep`; not in v1).
- Publishing models to Hugging Face Hub.

**Anti-goals: it fails if**
- A "winning" recipe is a lucky seed. Any keep decision must clear the noise floor.
- The agent can edit the trainer, the scorer, the data split or the test set, or peek at test
  labels before `/finetune-kit.promote`.
- The user must hand-tune ranges or values to get started.
- Experiment data leaves the machine (datasets may hold real personal data, as Enron did).

## 3. Constraints

- Python ≥ 3.12, PyTorch, Hugging Face `transformers`, `peft` (LoRA), installed and run via `uv`.
- Must work on Apple Silicon (MPS) as well as CUDA. Known MPS quirk: models with fp16
  checkpoints must load in fp32 (from the PII bake-off).
- No network during runs except the first model download. Model revisions are pinned per
  experiment so reruns are reproducible.
- Runs take 20–60 minutes. Agent shell tools time out well before that (Claude Code Bash caps
  at 10 minutes), so runs must detach and the agent must poll.
- One training job at a time per machine (memory: DeBERTa-v3-base peaked ~15.7 GB on M4).

## 4. Approaches considered

| Approach | Trade-off | Decision |
|---|---|---|
| Web UI + MCP server | Rich visuals, but a server and UI to build and run; agent wiring per tool | Rejected: user prefers terminal |
| CLI + `AGENTS.md` only | Least work, portable; no install/update story, rules are loose text | Folded in |
| Claude Code plugin only | Best Claude Code experience; excludes Codex and Cursor | Rejected |
| **Spec-kit-style installer + CLI** | One `init --ai <agent>` writes that agent's slash commands, shared templates and rules into the project; flow is explicit, staged, file-backed | **Chosen** (user's pick) |

Model: [github/spec-kit](https://github.com/github/spec-kit).

## 5. What the user experiences

```bash
uv tool install git+<repo>   # once per machine
finetune-kit init --ai claude                 # or: codex | cursor
```

Then, in the agent, eight commands run in order. Each is a prompt file for the agent, which
calls `finetune-kit` CLI commands (JSON output) as its tools. The agent drives the whole conversation,
asking questions one at a time.

| # | Command | Agent does | Writes (in `experiments/NNN-<name>/`) |
|---|---|---|---|
| 1 | `/finetune-kit.setup` | Asks: task (seq-cls / token-cls), dataset path(s), model (from `finetune-kit models`) or "bake-off" with 2–3 candidates, metric to optimize (from `finetune-kit metrics`), optional guardrail metric, seeds per run (default 3), budget (runs, hours) | `experiment.md`, `experiment.yaml` |
| 2 | `/finetune-kit.check` | Runs `finetune-kit check-data`; explains findings: format errors, label balance, token lengths vs max length, duplicates, leakage across splits, **validation too small for the metric** (e.g. `pauc_hi` with recall band 0.98 needs ≥ 500 validation positives so the band spans ≥ 10 positives; other metrics warn below 100 rows in the smallest class; suggests a bigger validation share); proposes fixes; creates splits if none given | `data-report.md`, `splits/` |
| 3 | `/finetune-kit.bakeoff` (optional; runs when setup chose bake-off) | Each candidate model at its profile defaults on N seeds; ranks by mean objective; a model wins only if it beats the runner-up by the noise bar, otherwise the faster one is recommended; user confirms; the experiment's model is fixed from here on | `bakeoff.md` |
| 4 | `/finetune-kit.hparams` | Shows the catalog for this task + model, grouped, with ★ recommendations derived from the data report and metric; user says which may be tuned (names only, no values) | `search-space.md`, `search-space.yaml` |
| 5 | `/finetune-kit.baseline` | If LR is tunable: LR scan. Then noise floor: profile defaults on N seeds → σ for objective and guardrail | `baseline.md` |
| 6 | `/finetune-kit.research` | Autoresearch loop under the research rules (§7) until budget or stop condition | `results.tsv`, `log.md` |
| 7 | `/finetune-kit.analyze` | Compares kept vs discarded, explains what helped, suggests next tunables; **error analysis**: shows the worst validation misses and false alarms of the best run (`finetune-kit misses`) and flags likely label problems | `report.md`, `report.html` |
| 8 | `/finetune-kit.promote` | Final confirm: chosen recipe on more seeds, scored once on test; saves model, threshold (if any), recipe; **measures throughput** (rows/s on this device and on CPU, model size) and **exports a standalone `predict.py`** | `final/` |

Users may rerun `/finetune-kit.hparams` then `/finetune-kit.research` to open more settings after a
plateau (as the PII sweep did across rounds r1–r3). Each command checks that the previous
stage's file exists and refuses otherwise.

### Example: `/finetune-kit.hparams`

```
Agent: For seq-cls on UltraBERT-150M, here is what can be tuned.
       ★ = recommended for your data (see data-report.md).
  Optimizer & schedule
   ★ learning rate        most sensitive setting
   ★ layer-wise LR decay  biggest single gain in past tuning
     warmup · schedule · weight decay · epochs · batch size · head LR multiplier
  Head:   pooling (cls/mean/attention) · head (linear/mlp) · dropout
  Loss
   ★ class weight         labels are 18% / 82%
   ★ focal loss
     CE + rank (good for AUC-style metrics) · label smoothing · R-Drop
  Regularization: EMA · re-init top layers · freeze embeddings · FGM · SWA
  Speed:  bf16 · max length
  Which may I change? Anything left out stays at the profile default.
```

## 6. Components

### 6.1 Installer (`finetune-kit init`)
Mirrors spec-kit: `--ai claude|codex|cursor`, `--here`. Writes:

```
.finetune-kit/
  memory/research-rules.md   # the "constitution" (§7); user-editable, versioned
  templates/                 # experiment.md, search-space.md, baseline.md, report.md …
  models.yaml                # project-local model profile overrides (optional)
<agent command files>        # one per stage, in that agent's format and location
experiments/                 # NNN-<name>/ folders, sequential numbering
```
Per-agent file locations follow spec-kit's current conventions (Claude Code skills,
Codex prompts/skills, Cursor commands). For Claude Code it also installs a `PreToolUse`
hook that blocks edits to `.finetune-kit/` locked files, `splits/` and `final/`.

### 6.2 `finetune-kit` CLI (the agent's tools)
All commands print JSON. Each validates its inputs and fails with a clear message.

| Command | Purpose |
|---|---|
| `finetune-kit models [--task]` | Model profiles available |
| `finetune-kit metrics [--task]` | Metric presets and their parameters |
| `finetune-kit check-data <exp>` | Data checks + split creation |
| `finetune-kit bakeoff <exp> --models a,b,c` / `finetune-kit bakeoff choose <exp> <model>` | Queue bake-off runs / fix the experiment's model |
| `finetune-kit misses <exp> <run_id> [--k 20]` | Worst validation misses and false alarms of a run |
| `finetune-kit catalog <exp>` | Hyperparameter/loss catalog for this task + model, with recommendations |
| `finetune-kit run <exp> --config JSON [--seeds N] [--stage lr-scan\|noise\|research]` | Validates config against `search-space.yaml` and hard limits, queues a run, returns `run_id` immediately |
| `finetune-kit wait <run_id> [--timeout S]` / `finetune-kit status` | Poll; returns per-seed metrics when done |
| `finetune-kit compare <exp>` | Table of runs: mean ± std, keep/discard verdict vs current best |
| `finetune-kit report <exp>` | HTML report with curves |
| `finetune-kit promote <exp> <run_id>` | Final confirm and export |

The trainer lives in the installed package, not in the project, so the agent never edits it.

**Distribution (my call).** Like spec-kit, installed from a git repo, not PyPI, in v1:
`uv tool install git+<repo>` provides the single `finetune-kit` command (installer + CLI, with torch) as one
global tool, separate from the user's project dependencies. Each experiment records the finetune-kit
version, torch/transformers versions and model revision in `experiment.yaml`; `finetune-kit run` warns
if they differ from what the baseline used.

### 6.3 Trainer
Generalized from `gate_autoresearch/sweep_train.py`: one JSON config, unknown keys rejected.

- **Tasks:** seq-cls (CLS / mean / attention / weighted-layer pooling → head, optional multi-sample dropout) and token-cls (per-token head,
  label alignment: first sub-token or all sub-tokens; optional CRF head).
- **Optimizer/schedule:** AdamW, lr, β2 (tunable; β1 0.9 and eps fixed), weight decay, warmup, linear/cosine/constant,
  LLRD, head LR multiplier, grad clip, batch + micro-batch (accumulation), epochs.
- **Losses:** CE, weighted CE (class weights), focal (γ), asymmetric loss (ASL: γ−, γ+, clip),
  label smoothing, pairwise rank, CE + rank, R-Drop. For token-cls: CE / weighted CE / focal /
  Dice (soft Dice over entity tags mixed with CE); CRF NLL when CRF head (not combined with Dice).
- **Regularization:** head dropout, backbone hidden and attention dropout, EMA, SWA, re-init top N layers, freeze embeddings, FGM.
- **Sampling:** minority oversampling factor (seq-cls).
- **Parameter-efficient:** LoRA (rank, alpha) on profile-defined target modules; off by default.
- **Precision:** fp32 / bf16 autocast.
- **Evaluation:** validation every half epoch; best-on-validation checkpoint kept in memory
  (as `pii_classifier/train_full.py` does).
- **Length:** `max_len` from profile; over-length rows are truncated and counted (count goes
  into run output); `check-data` warns beforehand.

### 6.4 Model profiles (`models.yaml`, shipped + overridable)
Per model: HF id, pinned revision, tasks supported, LR default and hard limits, max length,
safe batch/micro-batch per device class, quirks (`force_fp32_on_mps`, `grad_checkpointing`,
`trust_remote_code`), the config keys for backbone dropout, and LoRA target modules. v1 list: UltraBERT-150M, ModernBERT-base/large, DeBERTa-v3-base/large,
RoBERTa-base/large, BERT-base. Unknown HF ids get a generic profile and a mandatory LR scan.

UltraBERT-150M is `dmis-lab/UltraBERT-150M` (public on HF, pinned in the PII work at revision
`545ac9d9c7aece444f014ada5a1e9818132b4d35`), needs `trust_remote_code=True`, max length 512.
DeBERTa-v3-base is `microsoft/deberta-v3-base` @ `8ccc9b6f…`, fp16 checkpoint → force fp32 on MPS,
gradient checkpointing on. Remote-code models are allowed only from profiles shipped with
finetune-kit or added explicitly by the user (`finetune-kit models add --trust-remote-code`).

Seed values (from PII runs; others from model cards/papers, refined by the LR scan):
UltraBERT 8e-5 (4e-5 with LLRD 0.75), DeBERTa-v3-base 1e-5, RoBERTa-base 2e-5,
RoBERTa-large 1e-5, ModernBERT-base 5e-5.

### 6.5 Metrics
- seq-cls general: accuracy, F1, macro-F1, micro-F1, ROC-AUC, PR-AUC, log-loss.
- Recall-first (binary, or one positive class): `pauc_hi` with configurable recall band
  (default 0.98–1.0, as `gate_autoresearch/eval.py`), skip@recall, precision@recall.
- Precision-first: recall@precision.
- token-cls: entity-level F1 (seqeval), per-type F1, token accuracy.
- Custom: `metric(y_true, scores) -> float` in a user Python file, referenced from
  `experiment.yaml`.
Every metric declares its direction (higher/lower better) and its `min_delta`.

**Positive class and threshold (my call).** Recall-first and precision-first metrics need a
positive class: for binary it is label `1`; for multi-class `/finetune-kit.setup` asks which class.
At promote, if the objective is recall- or precision-constrained, the threshold is chosen on
validation at the metric's target (e.g. 0.995 recall) and saved in
`final/threshold.json`; otherwise prediction is argmax and no threshold is saved.
Custom metrics are seq-cls only in v1 and receive `y_true` (n,) and `probs` (n, C).

### 6.6 Data contract
- seq-cls JSONL: `{"id", "text", "label", "group_id"?}`. `label` int or string.
- token-cls JSONL: `{"id", "tokens": [...], "tags": [...]}` with BIO/BIOES tags, `group_id`?.
- Splits: user gives train/val(/test) files, or `check-data` splits one file 80/10/10,
  grouped by `group_id` (document) when present, stratified by label. Splits are frozen after
  `/finetune-kit.check`; test is only read by `finetune-kit promote`.

### 6.7 Run store
`experiments/NNN-<name>/runs/<run_id>/seed_<k>/{config.json, metrics.json, train.log}` plus a
row per run in `results.tsv` (`run_id, stage, objective_mean, objective_std, guardrail_mean,
n_seeds, minutes, verdict, description`). No checkpoints kept except in `final/` (disk).
Runs execute one at a time from a simple local queue (file-based is fine).

## 7. Research rules (`research-rules.md` + CLI enforcement)

Text rules for the agent, from `gate_autoresearch/program.md`, generalized:
1. Change one idea per run, so each result teaches something. Write a one-line description.
2. Only settings in `search-space.yaml` may change; all others stay at profile defaults.
3. **Keep** a run only if
   `objective_mean − best_mean > max(2·σ_obj, min_delta)` and
   `guardrail_mean ≥ kept_guardrail_mean − σ_guard`. Otherwise discard and go back to the
   last kept config. Equal result with a simpler config: keep.
4. Crash or timeout: discard, note the error, do not retry the same config unchanged.
5. Stop after 10 non-keeps in a row, or when the budget is spent; then write the summary.
6. Never read test data; never change splits, metrics or rules.

Enforced by the CLI regardless of what the agent does: search-space and hard-limit
validation, seeds per run, budget, the keep verdict (computed by `finetune-kit compare`, not the
agent), test isolation, and a checksum of `research-rules.md` / `search-space.yaml` /
`splits/` recorded at baseline and checked on every `finetune-kit run`.

σ comes from the noise-floor runs (std of the objective across N seeds of the default config).

**Statistics (my call).** Runs are compared as means over N seeds, so the noise term is the
standard error of a difference of two means: `noise = σ · sqrt(2/N)`. Keep if
`Δ > max(2 · noise, min_delta)` and the guardrail mean is not below the kept guardrail mean
by more than `σ_guard · sqrt(2/N)`. `min_delta` defaults to 0.002 for metrics in [0, 1] and is
set per metric in the metric registry (e.g. log-loss uses a relative 1%); user can override in
`experiment.yaml`. With N = 1 this reduces to `2·σ·√2`, slightly stricter than the PII rule.

**"Equal but simpler" (my call).** Computed by the CLI, not the agent: if `|Δ|` is within the
noise term and the config has fewer non-default keys than the kept one, verdict is
`keep-simpler`. The agent never overrides a verdict.

**Reopening the search space (my call).** Rerunning `/finetune-kit.hparams` starts a new *round*:
`finetune-kit space commit` (called by that stage, after the user confirms in the conversation) writes the
new `search-space.yaml`, records its checksum and round number. The current best config, σ and
budget carry over (same data, same defaults, so the noise floor still holds). `finetune-kit run` rejects a
`search-space.yaml` whose checksum doesn't match the last commit.

**Test isolation (my call).** The test split is not stored in the project. `check-data` moves it
to `~/.finetune-kit/vault/<experiment-id>/test.jsonl` (mode 600) and keeps only its hash and row count
in `splits/`. Only `finetune-kit promote` reads the vault. This works the same in all three agents, without
relying on hooks. Text rules say never to read `~/.finetune-kit/`.

**Long runs and resuming (my call).** `/finetune-kit.research` is designed to run unattended for hours
to days. All state is in files (`results.tsv`, `runs/`, queue), so if the agent session ends or
runs out of context, rerunning `/finetune-kit.research` reads `log.md` + `finetune-kit compare` and continues from
the last kept config. The queue survives restarts; a run that was in progress when the machine
stopped is marked `crash` and may be retried once.

To keep the loop fast, `/finetune-kit.setup` offers a **research subset** (e.g. 3,000 training rows,
as the PII smoke test used). Research runs train on the subset; `/finetune-kit.promote` retrains the
chosen recipe on the full training set before scoring test.

**Run length (my call).** Runs use fixed epochs (tunable when the user allows it). Each run's
timeout is 3× the baseline run's training time. bf16, max length and batch size change speed
but are compared on the same epochs; their speed gain is reported alongside the metric.

## 8. Decided by the user
- Terminal + agent, no web UI. Agent drives every conversation.
- Spec-kit-style installer and staged slash commands for Claude Code, Codex and Cursor.
- Tasks: sequence classification and token classification.
- User chooses the metric from a list; user chooses which hyperparameters and losses are
  tunable, by name only. The agent chooses values.
- LR defaults per model via profiles, refined by an automatic LR scan.
- A dedicated hyperparameter-selection stage (`/finetune-kit.hparams`).
- Asymmetric loss (seq-cls) and Dice loss (token-cls).
- An optional model bake-off stage, error analysis in analyze, a small-validation warning in
  check, throughput measurement and `predict.py` export at promote.
- Extra tunables: backbone dropout, weighted layer pooling, multi-sample dropout, Adam β2,
  minority oversampling, LoRA.

## 9. My calls (correct any of these)
- Name finetune-kit: command `finetune-kit`, distribution `finetune-kit`, import package `finetune_kit` (free on PyPI).
- Local compute only, one run at a time.
- Default 3 seeds per run; keep rule exactly as the PII `program.md`.
- Hard limits per hyperparameter exist but are invisible during selection; they only reject
  nonsense.
- Recommendations (★) in `/finetune-kit.hparams` are rule-based (imbalance → class weight/focal, recall-first metric → ASL, rare entities → Dice,
  AUC-style metric → rank loss, LR/LLRD always) rather than learned.
- Over-length inputs truncate with a count, rather than failing.
- No Optuna sweep mode in v1.
- Keep rule uses seed-mean statistics (`σ·√(2/N)`), and "equal but simpler" is decided by the CLI (§7).
- Rerunning `/finetune-kit.hparams` starts a new round; best config and σ carry over (§7).
- Test split lives in `~/.finetune-kit/vault/`, outside the project; only `finetune-kit promote` reads it (§7).
- Research runs unattended and resumes from files; optional research subset, full retrain at promote (§7).
- Fixed epochs per run; timeout 3× baseline (§7).
- Positive class asked for multi-class recall/precision metrics; threshold set on validation (§6.5).
- Installed from git with `uv tool install`, not PyPI (§6.2).

## 10. Left to the builder
- Exact per-agent command file formats and locations (follow current spec-kit).
- Queue implementation, process detaching and log streaming.
- `report.html` design.
- LR scan details (number of points, fraction of steps).
- Code layout inside the `finetune-kit` package, tests, and CI.
