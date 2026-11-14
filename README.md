# finetune-kit

Noise-aware autoresearch for BERT-family encoders. Your coding agent (Claude Code, Codex or
Cursor) acts as the researcher: it picks hyperparameters, queues runs, reads results and decides
what to try next. `finetune-kit` is its toolbox: a trainer, a run queue, a metric registry and a
keep rule that the agent cannot override.

**Why.** One fine-tuning run varies by several points from seed to seed, so a single "better"
run proves nothing. finetune-kit compares multi-seed means against a measured noise floor, keeps
the test set out of reach until the end, and leaves a reviewable file behind at every stage. The
output is a final recipe backed by multi-seed evidence, not a lucky seed.

Tasks: sequence classification (binary or multi-class) and token classification (NER). Runs
locally on CUDA, Apple MPS or CPU, one job at a time. Experiment data never leaves the machine.

## Install

```bash
uv tool install git+<repo>        # once per machine; provides the `finetune-kit` command
```

Requires Python 3.12 or newer. Torch and transformers are installed with the tool.

## Quick start

In your project directory:

```bash
finetune-kit init --ai <agent>
```

Pick the `--ai` value for the agent you use:

| Agent | Command |
|---|---|
| Claude Code | `finetune-kit init --ai claude` |
| Codex | `finetune-kit init --ai codex` |
| Cursor | `finetune-kit init --ai cursor` |

If you are an AI agent installing this for a user, choose the value for the agent you are
yourself; do not copy a value from an example.

Then open the agent and run the eight stages in order. Each stage is a prompt file the agent
follows; it calls `finetune-kit` commands (one JSON object per call) and writes a reviewable file.

| Stage | What it does |
|---|---|
| `setup` | Asks task, data, model (or bake-off), metric, seeds and budget; creates the experiment. |
| `check` | Validates data, warns about small validation sets, freezes splits, moves test to the vault. |
| `bakeoff` | Optional. Runs 2-3 candidate models at defaults and fixes the winner. |
| `hparams` | You name which hyperparameters and losses may be tuned; the agent picks the values. |
| `baseline` | Learning-rate scan, then the noise floor (default recipe on N seeds). |
| `research` | The autoresearch loop under the keep rule until the stop rule or budget. |
| `analyze` | What helped, error analysis of the best run (worst misses), `report.html`. |
| `promote` | Full retrain, test scored once, `final/` with model, recipe, throughput and `predict.py`. |

How a stage is invoked per agent (after `init`):

| Agent | Invocation | Installed under |
|---|---|---|
| Claude Code | `/finetune-kit-setup`, `/finetune-kit-check`, ... | `.claude/skills/` |
| Codex | `$finetune-kit-setup`, `$finetune-kit-check`, ... | `.agents/skills/` |
| Cursor | `/finetune-kit.setup`, `/finetune-kit.check`, ... | `.cursor/commands/` |

Each stage checks that the previous stage's file exists and refuses otherwise. Re-running
`hparams` then `research` opens a new round after a plateau; the best config, noise floor and
budget carry over. If the agent session ends, re-run `research`: all state is in files and the
loop continues from the last kept config.

Without an agent you can drive the same CLI by hand: `finetune-kit --help`, and `--help` on each
subcommand (`new`, `check-data`, `bakeoff`, `space commit`, `lr-scan`, `run`, `wait`, `status`,
`compare`, `misses`, `report`, `promote`, `models`, `metrics`, `catalog`).

## Data formats

Local files: `.jsonl`, `.csv`, `.tsv` and `.parquet`. CSV and TSV are for sequence classification
only. JSONL has one object per line.

```json
{"id": "r1", "text": "call me at 555 1234", "label": 1, "group_id": "doc7"}
```

Sequence classification: `id`, `text`, `label` (int or string); `group_id` optional.

```json
{"id": "t1", "tokens": ["Ann", "Lee", "lives", "here"], "tags": ["B-PER", "I-PER", "O", "O"], "group_id": "doc7"}
```

Token classification: `id`, `tokens`, `tags` (BIO or BIOES); `group_id` optional. A row without an
`id` gets a generated one (`train-000001`).

**Column mapping.** If your columns have other names, map them with `--columns`:

```bash
finetune-kit new demo --task seq-cls --model roberta-base --train reviews.csv \
    --columns '{"text": "review", "label": "sentiment"}' --metric macro_f1
```

Keys are `text`, `label`, `id`, `tokens`, `tags`, `group_id`; omit it when the names already match.
`finetune-kit inspect-data --train reviews.csv` (or `--source hf:...`) lists the columns, row counts,
label values and a suggested mapping without printing any row text.

**Hugging Face datasets.** Use `--source hf:<id>[:<config>]` instead of `--train`, optionally with
`--revision <branch|tag|sha>`; `new` pins the exact revision, so later stages see the same data.
Class-label integers become label names. The dataset files at the pinned revision are downloaded
into `~/.finetune-kit/hf-cache/` (not `~/.cache/huggingface/`), so a pinned dataset also loads offline.
Private or gated datasets use your existing Hugging Face login (`hf auth login` or
`huggingface-cli login`); finetune-kit never asks for or stores a token.

**How splits are made.**

- One file (`--train` only): `check-data` splits it 80/10/10, stratified by label and grouped by
  `group_id` when present. Or give `--train --val --test` yourself.
- Hugging Face: published `train`, `validation` and `test` splits are used as they are (`valid`,
  `val` and `dev` count as validation). A missing, empty or unlabelled validation or test split is
  carved from train (a split where only some rows lack labels is refused); with only `train`, it
  is split 80/10/10. `data-report.json` records `split_origin`, `ignored_splits` and a warning
  "<role> carved from train: <reason>" for each carved part.

Rows over the model's max length are truncated and counted; `check-data` warns beforehand.

## Where things live

- `experiments/NNN-name/`: one folder per experiment. `experiment.yaml` (config),
  `search-space.yaml`, `state.json`, `data-report.json`, `splits/` (train, val, test hash only),
  `runs/<run_id>/seed_<k>/` (config, metrics, log), `results.tsv`, `report.html`, and `final/`
  after promote (`model/`, `recipe.json`, `test_metrics.json`, `throughput.json`, `predict.py`).
  The stage files (`experiment.md`, `search-space.md`, `baseline.md`, `log.md`, `report.md`) sit
  next to them.
- `.finetune-kit/`: `memory/research-rules.md` (the rules the agent follows, user-editable),
  `models.yaml` (project-local model profiles), and the templates.
- `~/.finetune-kit/vault/<experiment-id>/test.jsonl`: the test split, mode 600, outside the
  project. Only `finetune-kit promote` reads it. The same directory holds the run queue and
  worker log. Set `FINETUNE_KIT_HOME` to move it.

## The keep rule

A run's score is its mean over N seeds. The noise floor sigma is the std of the objective across
N seeds of the default config, measured once at `baseline`. A research run is kept only if its
objective beats the current best by more than `max(2 * sigma * sqrt(2/N), min_delta)` (direction
aware) and the guardrail metric, if set, is not worse than the kept guardrail by more than
`sigma_guard * sqrt(2/N)`. If the difference is within the noise and the config has fewer
non-default keys than the kept one, the verdict is `keep-simpler`. Otherwise it is discarded and
the next idea starts from the last kept config. Ten non-keeps in a row, or an exhausted budget,
stop the loop. The verdict is computed by `finetune-kit compare` and the worker, never by the agent.

## Metrics, losses and tunables

Metrics (`finetune-kit metrics`; each declares its direction and `min_delta`):

- seq-cls general: `accuracy`, `f1`, `macro_f1`, `micro_f1`, `roc_auc`, `pr_auc`, `log_loss`.
- Recall-first: `pauc_hi` (recall band, default 0.98-1.0), `skip_at_recall`, `precision_at_recall`.
- Precision-first: `recall_at_precision`.
- token-cls: `entity_f1`, `token_accuracy`.
- Custom seq-cls metric: `--custom-metric PATH:FUNC --direction max|min`, with
  `func(y_true, probs) -> float`.

Losses: seq-cls `ce`, `focal`, `asl`, `rank`, `ce+rank`; token-cls `ce`, `focal`, `dice`.

Tunable hyperparameters (`finetune-kit catalog EXP` shows them with recommendations):

- Optimizer and schedule: `lr`, `llrd`, `head_lr_mult`, `weight_decay`, `beta2`, `clip`, `warmup`,
  `schedule`, `epochs`, `batch`.
- Head: `pooling`, `pool_layers`, `head`, `crf` (token-cls).
- Loss settings: `loss`, `class_weight`, `focal_gamma`, `label_smoothing`, `rank_weight`,
  `asl_gamma_neg`, `asl_gamma_pos`, `asl_clip`, `dice_smooth`, `dice_ce_weight`.
- Regularization: `dropout`, `hidden_dropout`, `attn_dropout`, `msd_samples`, `rdrop_alpha`, `ema`,
  `swa`, `reinit_top`, `freeze_embed`, `fgm_eps`.
- Adapter, data, system: `lora_r`, `lora_alpha`, `oversample`, `max_len`, `label_align`, `precision`.

You choose which are tunable, by name; anything left out stays at the profile default. Hard limits
exist per hyperparameter and only reject nonsense.

## Model profiles

`finetune-kit models` lists them: `ultrabert-150m`, `deberta-v3-base`, `deberta-v3-large`,
`roberta-base`, `roberta-large`, `modernbert-base`, `modernbert-large`, `bert-base`. A profile pins
the Hugging Face revision and sets the LR default and range, max length, batch sizes, quirks (fp32 on
MPS, gradient checkpointing, remote code), dropout config keys and LoRA targets.

To add a model: `finetune-kit models add <hf_id> [--trust-remote-code]` writes a generic profile
into `.finetune-kit/models.yaml` (it reads only `config.json`). Unknown models get a generic
profile, so run the LR scan before trusting the default LR. Edit `.finetune-kit/models.yaml` to
refine it. An entry there for a shipped key overrides only the fields it lists (for example
`roberta-base: {lr_default: 3e-5}`); a new key must list every required field. `models add`
refuses a key that already exists. Remote-code models are allowed only from shipped profiles or
when you add them explicitly with `--trust-remote-code`.

## Trust and safety

- **Test isolation.** `check-data` moves the test split to the vault outside the project and keeps
  only its hash and row count. Only `promote` reads it, once. `misses` reads validation only.
- **Tamper checks.** Checksums of `research-rules.md` and the train/val splits (recorded by
  `check-data`), `search-space.yaml` (at each commit), and `experiment.yaml` and the resolved model
  profile (at the first noise run) are verified on every `finetune-kit run`, again when a run
  starts, and at `promote`; a mismatch refuses the run, and `wait` reports the refusal. `init
  --force` replaces an edited `research-rules.md`, so existing experiments then fail this check.
- **Enforced by the CLI, not the agent:** search-space and hard-limit validation, seeds per run,
  budget, run timeout (3x baseline), and the keep verdict.
- **Guard hook (Claude Code only).** `init --ai claude` installs a `PreToolUse` hook that blocks
  edits to the locked files (rules, splits, search space, `experiment.yaml`, `data-report.json`,
  state, results, runs, `final/`). Codex and Cursor get the rules in text only, with no hook.
- **Limits.** A determined agent outside Claude Code could still edit state files directly or read
  `~/.finetune-kit/`. The tamper checks catch edits to protected files at the next `run`, but not
  every state file. Review `results.tsv` and `log.md` as you would any agent's work.

## Known issues

- The stage prompts have not yet been tried in a live agent session (Claude Code, Codex or
  Cursor); expect rough edges in the conversation flow.
- `finetune-kit init` prints plain text, not a JSON object like the other commands.
- Runs are local and sequential; there is no remote GPU queue, web UI or multi-label support.
