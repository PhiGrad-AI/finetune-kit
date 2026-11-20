# finetune-kit

**Train your own text classifier by answering questions.** You bring examples (for instance
emails labelled "spam" or "not spam"), your AI coding agent asks you a few plain questions, runs
the experiments on your computer and tells you what worked. At the end you get a trained model
and a one-line command to use it on new text.

You don't need to know machine learning. Every choice is explained as it comes up, with a
recommendation, and you can always answer "use the recommended one".

Works with **Claude Code**, **Codex** and **Cursor**. Your data never leaves your machine.

## What it can do

- **Sort text into categories** ("sequence classification"): spam or not, which department a
  ticket belongs to, positive or negative review, prompt injection or safe.
- **Find things inside text** ("token classification", also called NER): names, places,
  phone numbers, product codes.

## What you need

- A Mac or Linux computer (Windows is untested). An NVIDIA GPU or an Apple Silicon Mac (M1 or later) makes
  training much faster; a plain CPU works but is slow (see [How long it takes](#how-long-it-takes)).
- One of the AI agents above, installed and signed in.
- `uv`, a Python installer. If you don't have it:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

  You don't need to install Python yourself; uv fetches Python 3.12 if needed.
- `git`, which uv uses to download finetune-kit from GitHub (no GitHub account needed).
- Your examples, as a CSV, JSONL or Parquet file, or the name of a Hugging Face dataset
  (see [Your data](#your-data)).

## Quick start

**1. Install** (once per computer):

```bash
uv tool install "git+https://github.com/PhiGrad-AI/finetune-kit"
```

**2. Set up a project folder.** Make a folder for this work, open a terminal in it and run the
line for your agent:

| Agent | Command |
|---|---|
| Claude Code | `finetune-kit init --ai claude` |
| Codex | `finetune-kit init --ai codex` |
| Cursor | `finetune-kit init --ai cursor` |

If you are an AI agent installing this for a user, choose the value for the agent you are
yourself; do not copy a value from an example.

**3. Start.** Open your agent in that folder and type:

| Agent | Type this |
|---|---|
| Claude Code | `/finetune-kit-start` |
| Codex | `$finetune-kit-start` |
| Cursor | `/finetune-kit.start` |

The agent lists the steps, then asks the first question. From here on you just answer.

## Your first run: a worked example

Want to try it before using your own data? This uses a public dataset of movie reviews labelled
positive or negative (about 10,000 short reviews). Answer the steps like this:

| Step | What to say |
|---|---|
| Choose the task | "Sort text into categories" |
| Choose the data | "Use the Hugging Face dataset `cornell-movie-review-data/rotten_tomatoes`" |
| Choose the model | "Use the recommended one" |
| Choose the goal | "Accuracy" |
| Check the data | "Looks good, continue" |
| Choose the loss | "ce and the starred ones" |
| Choose the settings | "Use the recommended ones" |
| Choose the learning speed | "Use the default" |
| Choose how much to try | "10 experiments, 2 hours", and "yes" to training on a sample |
| Run the experiments | "Go" |

Then come back later, ask for the report, and finish.

## The steps

The agent walks you through these one at a time. Each step is one small decision, and each one
ends by telling you the next step and its command.

| # | Step (command suffix) | What you decide |
|---|---|---|
| 1 | Choose the task (`choose-task`) | Say what kind of problem the model should learn to solve |
| 2 | Choose the data (`choose-data`) | Point to the examples the model will learn from |
| 3 | Choose the model (`choose-model`) | Pick the model to start from, and optionally others to compare it with |
| 4 | Choose the goal (`choose-goal`) | Say what counts as a good result and how it is scored |
| 5 | Check the data (`check-data`) | Look the data over for problems, then set aside examples to test with |
| 6 | Compare models (`compare-models`) | Try the models you picked side by side and keep the best one |
| 7 | Choose the loss (`choose-loss`) | Pick the losses (ways of measuring mistakes) to experiment with; each gets tried |
| 8 | Choose the settings (`choose-hyperparameters`) | Pick which training settings research may change |
| 9 | Choose the learning speed (`choose-learning-rate`) | Use the model's usual learning speed, or run a short test to find a better one |
| 10 | Choose how much to try (`choose-experiments`) | Decide how many experiments to run and how long they may take |
| 11 | Run the experiments (`start-experiment`) | Measure the starting point, then let research try improvements |
| 12 | Check progress (`status`) | See how the experiments are going at any time |
| 13 | Read the report (`report`) | See what was tried and what worked best |
| 14 | Finish (`finish`) | Train and score the final model, then wrap up |

Step 6 only applies if you picked models to compare in step 3. Step 12 is optional and can be
used at any time.

Each step ends by offering the next one, with its command (for example
`/finetune-kit-choose-data`), and asks whether to continue in the same conversation. The agent
never starts a step without your yes and never skips one; if you say no, type the command
whenever you like. If a step is started too early, it hands you to the step that comes first. If
the session ends, type `/finetune-kit-start` again: the agent picks up where the files say you
are. To try another round after the report (new losses or settings), start again from step 7;
the starting point, learning speed and budget carry over.

Each step ends by offering the next one, with its command (for example
`/finetune-kit-choose-data`), and asks whether to continue in the same conversation. The agent
never starts a step without your yes and never skips one; if you say no, type the command
whenever you like. If a step is started too early, it hands you to the step that comes first.

**Taking a break.** You can close the agent at any time. Everything is saved in files; type the
start command again and the agent picks up where you left off.

**Trying more.** After the report, you can start another round with other losses or settings:
start again from step 7. The best result so far, the learning speed and the remaining budget
carry over.

Where the commands are installed (after `init`):

| Agent | Invocation | Installed under |
|---|---|---|
| Claude Code | `/finetune-kit-<step>` | `.claude/skills/` |
| Codex | `$finetune-kit-<step>` | `.agents/skills/` |
| Cursor | `/finetune-kit.<step>` | `.cursor/commands/` |

## Your data

The simplest form is a CSV file with one example per row: a column with the text and a column
with its label.

```csv
text,label
"Win a free cruise, click now",spam
"Can we move the meeting to 3pm?",not spam
```

- **Columns named differently?** That's fine; the agent looks at your file and asks which column
  is which. It reads only column names and counts, never your rows.
- **How many examples?** A few hundred per label is a reasonable minimum; a few thousand is better.
- **Already split into train / validation / test?** Give all three and they are used as they are.
  Otherwise the agent splits one file for you (80% to learn from, 10% to check progress, 10% kept
  aside for the final score).
- **Hugging Face datasets** work too: give the name, such as `cornell-movie-review-data/rotten_tomatoes`.
  The dataset's own train, validation and test splits are used. For a private dataset, sign in
  first with `hf auth login`.
- **Finding things inside text** needs a different shape (a list of words and a tag for each); see
  [Data formats](#data-formats).

## How long it takes

Each experiment trains the model a few times (once per seed) to make sure a result is real and
not luck. Rough guide for a base-size model on a few thousand examples:

- **NVIDIA GPU:** a few minutes per experiment.
- **Apple Silicon Mac:** several minutes to tens of minutes per experiment.
- **CPU only:** an hour or more per experiment. Use a sample and a small budget, or a GPU.

In the *Choose how much to try* step you set a limit on the number of experiments and hours, and
on large datasets the agent offers to experiment on a sample to go faster. Research also stops on
its own when ten experiments in a row bring no real improvement. Experiments keep running while
you do other things; check on them with the *Check progress* step.

## What you get at the end

The *Finish* step trains the final model, scores it once on the examples kept aside, and puts
everything in `experiments/<your-experiment>/final/`:

- `model/`: the trained model.
- `predict.py`: a small script to use it. Put new rows in a JSONL file (one `{"text": "..."}` per
  line) and run:

  ```bash
  python predict.py --input new_rows.jsonl --output scores.jsonl
  ```

  Each output row has the predicted `label` and a `score`. It needs only `torch` and
  `transformers`, not finetune-kit, so you can copy the `final/` folder anywhere.
- `test_metrics.json`: the final score on the examples the model never saw.
- `recipe.json`: the exact settings that won, so the result can be reproduced.

The report (`report.html`, step 13) shows every experiment that was tried and why the winner won.
Open it in any browser.

## Troubleshooting

- **The agent says to "open Claude Code" but you use Codex or Cursor.** The project was set up for
  the wrong agent. Run `finetune-kit init --ai codex` (or `--ai cursor`) in the folder again.
- **The step commands don't appear in the agent.** Make sure you ran `init` in the same folder the
  agent has open, then restart the agent.
- **`uv tool install` says git is not found.** Install git (on a Mac, run `xcode-select --install`),
  then run the install line again.
- **Updating to the latest version.** Run the install line again with `--force` added after
  `install`, then run `init` again in your project.
- **The session ended in the middle.** Type the start command again; it continues from the files.
- **Stopping the experiments.** Interrupt the agent (Esc or Ctrl+C). The experiment already
  running finishes; no new ones start. Start *Run the experiments* again later to continue.
- **A Hugging Face dataset won't load.** For private or gated datasets, sign in with
  `hf auth login` first. finetune-kit never asks for your token.

## Words you'll see

- **Model:** a pre-trained language model (like RoBERTa or DeBERTa) that already understands
  language; training teaches it your labels.
- **Goal (metric):** the score that decides which experiment is best, such as accuracy.
- **Loss:** how training measures the model's mistakes. Some losses handle rare labels better.
  You pick several to compare; each gets tried.
- **Settings (hyperparameters):** knobs on training, like how many passes over the data or how
  large each batch is. You pick which ones research may change.
- **Learning rate (learning speed):** how big a step the model takes each time it learns; the most
  important setting.
- **Seed:** a random starting point. The same experiment with different seeds gives slightly
  different scores, so each experiment runs several seeds and uses the average.
- **Noise floor:** how much scores move from luck alone. An experiment only counts as better if
  it beats the best by more than this.
- **Validation / test examples:** validation examples are used to compare experiments; test
  examples are locked away and used once, at the end, for the honest final score.

---

## For engineers

Everything below is reference material; you don't need it to use finetune-kit.

One fine-tuning run varies by several points from seed to seed, so a single "better"
run proves nothing. finetune-kit compares multi-seed means against a measured noise floor, keeps
the test set out of reach until the end, and leaves a reviewable file behind at every step. The
output is a final recipe backed by multi-seed evidence, not a lucky seed.

Runs locally on CUDA, Apple MPS or CPU, one job at a time.

### The CLI

Each step prompt calls `finetune-kit` commands (one JSON object per call) and writes a reviewable
file, so you can drive the same flow by hand.

- `finetune-kit new NAME --task seq-cls|token-cls` creates a draft experiment; the answers go in
  one at a time: `set data EXP --train FILE` (or `--source hf:...`), `set model EXP MODEL
  [--compare MODEL,b[,c]]`, `set goal EXP --metric M [--guardrail M]`, then `check-data EXP`. The older
  one-shot form (`new` with `--model`, `--train`, `--metric`) still works.
- `set losses EXP ce,focal`, then `space commit EXP --tunable lr,llrd` (a first-round commit with
  no `--losses` uses the ones from `set losses`), `set learning-rate EXP default|scan`, and
  `set experiments EXP [--max-runs N] [--max-hours H] [--seeds N] [--subset N]`. On a big
  dataset `--subset`/`--seeds` alone are asked before the first training (model comparison or
  learning-rate scan) and are fixed once it has run; the budget is asked later.
- `run`, `wait`, `compare`, `misses`, `report` and `promote` run and read the experiments.
- `finetune-kit steps [EXP]` prints every step with its status (`done`, `next`, `to do`,
  `optional`, `skipped`), its number and its plain-language title.
- `finetune-kit next [EXP]` prints `{"experiment", "stage", "reason", "alternatives": [{"stage",
  "reason"}]}`, computed from the experiment's files, so it works without an agent. `stage` is a
  step name from the table. After `finish` it returns `choose-task` (the experiment is done;
  start a new one). With `--compare`, `next` names `compare-models` until a model is chosen, and
  skips it otherwise.

Each step checks that what it needs exists and refuses otherwise. Choosing new losses and
settings opens a new round after a plateau; the best config, noise floor and budget carry over.
All state is in files, so an interrupted research loop continues from the last kept config.

`finetune-kit --help` lists every subcommand (`new`, `set`, `steps`, `next`, `check-data`,
`bakeoff`, `space commit`, `lr-scan`, `run`, `wait`, `status`, `compare`, `misses`, `report`,
`promote`, `models`, `metrics`, `catalog`); `--help` on each shows its options.

### Data formats

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
`"id": null` ignores an `id` column and generates ids instead; a used `id` column must have no empty values.
`finetune-kit inspect-data --train reviews.csv` (or `--source hf:...`) lists the columns, row counts,
label values and a suggested mapping without printing any row text.

**Hugging Face datasets.** Use `--source hf:<id>[:<config>]` instead of `--train`, optionally with
`--revision <branch|tag|sha>`; `new` pins the exact revision, so later steps see the same data.
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

### Where things live

- `experiments/NNN-name/`: one folder per experiment. `experiment.yaml` (config),
  `search-space.yaml`, `state.json`, `data-report.json`, `splits/` (train, val, test hash only),
  `runs/<run_id>/seed_<k>/` (config, metrics, log), `results.tsv`, `report.html`, and `final/`
  after `finish` (`model/`, `recipe.json`, `test_metrics.json`, `throughput.json`, `predict.py`).
  The step files (`experiment.md`, `search-space.md`, `baseline.md`, `log.md`, `report.md`) sit
  next to them.
- `.finetune-kit/`: `memory/research-rules.md` (the rules the agent follows, user-editable),
  `models.yaml` (project-local model profiles), and the templates.
- `~/.finetune-kit/vault/<experiment-id>/test.jsonl`: the test split, mode 600, outside the
  project. Only the final training (`finetune-kit promote`, run by the `finish` step) reads it. The same directory holds the run queue and
  worker log. Set `FINETUNE_KIT_HOME` to move it.

### The keep rule

A run's score is its mean over N seeds. The noise floor sigma is the std of the objective across
N seeds of the default config, measured once when the experiments start (`start-experiment`). A research run is kept only if its
objective beats the current best by more than `max(2 * sigma * sqrt(2/N), min_delta)` (direction
aware) and the guardrail metric, if set, is not worse than the kept guardrail by more than
`sigma_guard * sqrt(2/N)`. If the difference is within the noise and the config has fewer
non-default keys than the kept one, the verdict is `keep-simpler`. Otherwise it is discarded and
the next idea starts from the last kept config. Ten non-keeps in a row, or an exhausted budget,
stop the loop. The verdict is computed by `finetune-kit compare` and the worker, never by the agent.

### Metrics, losses and tunables

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

### Model profiles

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

### Trust and safety

- **Test isolation.** `check-data` moves the test split to the vault outside the project and keeps
  only its hash and row count. Only the final scoring (`promote`, run by `finish`) reads it, once. `misses` reads validation only.
- **Tamper checks.** Checksums of `research-rules.md` and the train/val splits (recorded by
  `check-data`), `search-space.yaml` (at each commit), and `experiment.yaml` and the resolved model
  profile (at the first noise run) are verified on every `finetune-kit run`, again when a run
  starts, and at the final scoring (`promote`); a mismatch refuses the run, and `wait` reports the refusal. `init
  --force` replaces an edited `research-rules.md`, so existing experiments then fail this check.
- **Enforced by the CLI, not the agent:** search-space and hard-limit validation, seeds per run,
  budget, run timeout (3x baseline), and the keep verdict.
- **Guard hook (Claude Code only).** `init --ai claude` installs a `PreToolUse` hook that blocks
  edits to the locked files (rules, splits, search space, `experiment.yaml`, `data-report.json`,
  state, results, runs, `final/`). Codex and Cursor get the rules in text only, with no hook.
- **Limits.** A determined agent outside Claude Code could still edit state files directly or read
  `~/.finetune-kit/`. The tamper checks catch edits to protected files at the next `run`, but not
  every state file. Review `results.tsv` and `log.md` as you would any agent's work.

### Known issues

- `finetune-kit init` prints plain text, not a JSON object like the other commands.
- Runs are local and sequential; there is no remote GPU queue, web UI or multi-label support.
