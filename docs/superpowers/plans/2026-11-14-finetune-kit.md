# finetune-kit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `finetune-kit`, a CLI plus spec-kit-style agent commands that let Claude Code, Codex or Cursor run noise-aware fine-tuning research on BERT-family encoders for sequence and token classification.

**Architecture:** One Python package (`finetune-kit`, command `finetune-kit`) holds everything that must not be editable by the agent: data checks, metrics, model profiles, the hyperparameter catalog, the trainer, a file-based run queue with a detached worker, the keep/discard statistics and promotion. `finetune-kit init --ai <agent>` writes eight stage prompts (setup → check → bakeoff (optional) → hparams → baseline → research → analyze → promote) in that agent's format plus `.finetune-kit/` rules and templates. All state lives in `experiments/NNN-<name>/` files; the test split lives in `~/.finetune-kit/vault/`.

**Tech Stack:** Python ≥ 3.12, uv, typer, pydantic v2, PyYAML, numpy, scikit-learn, seqeval, torch ≥ 2.3, transformers ≥ 5.20, peft, pytorch-crf, pytest.

**Spec:** `docs/superpowers/specs/2026-11-14-finetune-kit-design.md`

## Global Constraints

- Package name `finetune-kit`; console script `finetune-kit` → `finetune_kit.cli:app`; import package `finetune_kit` under `src/finetune_kit/`.
- Installed with `uv tool install git+<repo>`; not published to PyPI.
- Every `finetune-kit` command except `init` and `report` prints exactly one JSON object to stdout; errors print `{"error": "<message>"}` and exit code 2.
- Devices: CUDA, else MPS, else CPU. Profiles with `force_fp32_on_mps: true` load in fp32 on MPS.
- No network during runs; model download only on first use. Model revision pinned per profile and recorded in `experiment.yaml`.
- Default seeds per run: 3. Default `min_delta`: 0.002 for metrics in [0, 1].
- Keep rule: `noise = σ·sqrt(2/N)`; keep iff `Δ > max(2·noise, min_delta)` and guardrail drop ≤ `σ_guard·sqrt(2/N)`; `keep-simpler` iff `|Δ| ≤ 2·noise` and fewer non-default keys; otherwise `discard`.
- Stop rule (agent text + `finetune-kit compare` flag): 10 non-keeps in a row, or budget spent.
- Run timeout: 3× the noise-floor run's mean training seconds per seed.
- Test split is stored only at `~/.finetune-kit/vault/<experiment_uid>/test.jsonl` (mode 600); only `finetune-kit promote` reads it, once.
- One training process at a time per machine.
- Slash command names: Claude Code `/finetune-kit-<stage>`, Codex `$finetune-kit-<stage>` skill, Cursor `/finetune-kit.<stage>` (stages: setup, check, bakeoff, hparams, baseline, research, analyze, promote).

## Review Focus

1. **String or mixed-type labels, or a label present in val/test but not train** → `check-data` maps labels to ids from train, reports unseen labels as an error, never crashes in training. Test in Task 3.
2. **Agent passes a value equal to the default for a non-tunable key, or a number as a string (`"4e-5"`)** → default-valued keys are accepted; strings are rejected with a message naming the key. Tests in Task 4 (validator) and Task 9 (CLI).
3. **Machine sleeps or the worker is killed mid-run** → on next worker start, a `running` run with a dead PID becomes `crash`, is retried once, and the queue continues. Test in Task 9.
4. **`finetune-kit init` in a project that already has `.claude/settings.json`, skills or `.finetune-kit/`** → merges hooks, overwrites only `finetune-kit-*` files, never deletes user files or experiments. Test in Task 12.
5. **Token-classification row longer than `max_len`** → truncation keeps labels aligned to surviving tokens and the truncated-row count is reported. Test in Task 7.

---

## File Structure

```
pyproject.toml
src/finetune_kit/
  __init__.py            __version__
  cli.py                 typer app; every command; JSON printing
  paths.py               experiment dir lookup, vault dir, worker lock path
  experiment.py          ExperimentConfig, SearchSpace, State models; load/save; checksums; space commit
  data.py                load/validate rows, label maps, check report, splitting, vault
  metrics.py             Metric registry and implementations
  profiles.py            ModelProfile, load_profiles(); profiles/models.yaml
  catalog.py             Hyperparameter catalog, hard limits, recommend(), validate_config()
  stats.py               RunSummary, verdict(), bakeoff_rank()
  store.py               run dirs, results.tsv, best/streak tracking
  queue.py               enqueue(), worker loop, stale-run recovery, timeouts
  train/
    config.py            TrainConfig (pydantic) with all trainer keys and defaults
    optim.py             param_groups(), lr_factor()
    losses.py            seq_loss(), token_loss()
    models.py            SeqClassifier, TokenClassifier, encode/align helpers
    loop.py              train_and_predict()
    worker.py            `python -m finetune_kit.train.worker` one-seed entry point
  promote.py             promote(), throughput benchmark, predict.py export
  report.py              HTML report
  installer.py           init(); agent writers; settings merge
  guard.py               PreToolUse hook entry (`finetune-kit _guard`)
  templates/
    commands/{setup,check,bakeoff,hparams,baseline,research,analyze,promote}.md
    predict.py.tmpl        standalone inference script exported at promote
    memory/research-rules.md
    docs/{experiment,search-space,baseline,report}.md
tests/
  conftest.py            tiny fixtures: seq-cls and token-cls jsonl, tiny model profile
  test_*.py              one per module
```

Tiny test models: profiles `tiny-bert` → `hf-internal-testing/tiny-random-bert` and `tiny-roberta` → `hf-internal-testing/tiny-random-RobertaModel` (registered only when env `FINETUNE_KIT_TEST_PROFILES=1`). Tests that need it are marked `@pytest.mark.model` and download once.

---

### Task 1: Package scaffold and CLI skeleton

**Files:**
- Create: `pyproject.toml`, `src/finetune_kit/__init__.py`, `src/finetune_kit/cli.py`, `src/finetune_kit/paths.py`, `tests/conftest.py`, `tests/test_cli.py`, `.gitignore`

**Interfaces:**
- Produces: `finetune_kit.cli.app: typer.Typer`; `finetune_kit.cli.emit(obj: dict) -> None` (prints JSON); `finetune_kit.cli.fail(msg: str) -> NoReturn` (prints `{"error": msg}`, exit 2); `finetune_kit.paths.kit_home() -> Path` (`$FINETUNE_KIT_HOME` or `~/.finetune-kit`); `finetune_kit.paths.find_experiment(name_or_path: str, root: Path = Path.cwd()) -> Path` (accepts `001`, `001-demo-exp` or a path; matches under `root/experiments/`); `finetune_kit.paths.vault_dir(uid: str) -> Path`.

- [ ] **Step 1:** `git init` in `autoresearch-ui/`; `.gitignore` with `.venv/ __pycache__/ experiments/ .finetune-kit/vault/`.
- [ ] **Step 2: Write failing tests** in `tests/test_cli.py`:

```python
def test_version_prints_json(runner):
    r = runner.invoke(app, ["version"])
    assert r.exit_code == 0 and json.loads(r.stdout)["version"] == finetune_kit.__version__

def test_find_experiment_by_number(tmp_path):
    (tmp_path / "experiments/001-pii").mkdir(parents=True)
    assert find_experiment("001", tmp_path).name == "001-pii"

def test_find_experiment_missing_fails(tmp_path):
    with pytest.raises(FileNotFoundError): find_experiment("009", tmp_path)

def test_kit_home_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FINETUNE_KIT_HOME", str(tmp_path)); assert kit_home() == tmp_path
```
`conftest.py` provides `runner = typer.testing.CliRunner()` and sets `FINETUNE_KIT_HOME` to a tmp dir for every test (autouse).
- [ ] **Step 3:** Run `uv run pytest tests/test_cli.py -v` → FAIL (import errors).
- [ ] **Step 4:** Implement. `pyproject.toml`: hatchling build, `[project.scripts] finetune-kit = "finetune_kit.cli:app"`, deps from Tech Stack, dev group pytest; pytest marker `model`.
- [ ] **Step 5:** Run tests → PASS. `uv run finetune-kit version` prints `{"version": "0.1.0"}`.
- [ ] **Step 6:** Commit `feat: package scaffold and finetune-kit CLI skeleton`.

---

### Task 2: Metrics registry

**Files:**
- Create: `src/finetune_kit/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Metric: name: str; tasks: tuple[str, ...]; direction: Literal["max","min"]; min_delta: float; params: dict; needs_positive: bool; constrained: Literal["recall","precision"] | None; fn: Callable`
  - seq-cls `fn(y: np.ndarray[int] (n,), probs: np.ndarray (n,C), params: dict, pos: int) -> float`
  - token-cls `fn(true_tags: list[list[str]], pred_tags: list[list[str]], params: dict) -> float`
  - `get(name: str) -> Metric`; `list_for(task: str) -> list[Metric]`; `compute(name, params, task, y_or_tags, probs_or_pred, pos=1) -> float` (merges `params` over defaults)
  - `threshold_for(metric: Metric, y, probs, params, pos) -> float | None` (recall: largest t with recall ≥ target; precision: smallest t with precision ≥ target; else None)
  - `load_custom(path: Path, func: str) -> Metric` (seq-cls only, direction from `experiment.yaml`)
- Names: seq-cls `accuracy, f1, macro_f1, micro_f1, roc_auc, pr_auc, log_loss, pauc_hi, skip_at_recall, precision_at_recall, recall_at_precision`; token-cls `entity_f1, entity_f1_<TYPE>` (via `entity_f1` with `params.type`), `token_accuracy`.
- `pauc_hi` defaults `{recall_min: 0.98, grid: 2001}`; `skip_at_recall` and `precision_at_recall` default `{recall: 0.995}`; `recall_at_precision` default `{precision: 0.95}`. `log_loss` is `min` with `min_delta` 0.01 × baseline treated as absolute 0.005. Binary metrics use score = `probs[:, pos]`; multi-class `f1` raises unless binary.

- [ ] **Step 1: Write failing tests** (values worked by hand; `pauc_hi`/`skip` semantics: skip counts all rows below t):

```python
def P(s): s = np.asarray(s); return np.stack([1 - s, s], 1)

def test_pauc_hi_perfect():
    assert compute("pauc_hi", {}, "seq-cls", np.array([0,0,1,1]), P([.1,.2,.8,.9])) == 1.0

def test_pauc_hi_overlap():
    assert compute("pauc_hi", {}, "seq-cls", np.array([0,0,1,1]), P([.1,.9,.5,.95])) == pytest.approx(0.5)

def test_skip_at_recall():
    assert compute("skip_at_recall", {}, "seq-cls", np.array([0,0,1,1]), P([.1,.9,.5,.95])) == pytest.approx(0.25)

def test_precision_at_recall():
    assert compute("precision_at_recall", {}, "seq-cls", np.array([0,0,1,1]), P([.1,.9,.5,.95])) == pytest.approx(2/3)

def test_positive_class_multiclass():
    y = np.array([0, 2, 2, 1]); probs = np.eye(3)[[0, 2, 2, 1]]
    assert compute("pauc_hi", {}, "seq-cls", y, probs, pos=2) == 1.0

def test_entity_f1():
    assert compute("entity_f1", {}, "token-cls", [["B-PER","I-PER","O"]], [["B-PER","I-PER","O"]]) == 1.0
    assert compute("entity_f1", {"type": "LOC"}, "token-cls", [["B-LOC","O"]], [["O","O"]]) == 0.0

def test_directions():
    assert get("log_loss").direction == "min" and get("pauc_hi").direction == "max"

def test_threshold_recall():
    t = threshold_for(get("skip_at_recall"), np.array([0,0,1,1]), P([.1,.9,.5,.95]), {}, 1)
    assert t == pytest.approx(0.5)

def test_unknown_metric():
    with pytest.raises(KeyError, match="unknown metric"): get("nope")

def test_custom_metric(tmp_path):
    (tmp_path/"m.py").write_text("def acc(y, probs): return float((probs.argmax(1)==y).mean())")
    m = load_custom(tmp_path/"m.py", "acc"); assert m.fn(np.array([1]), P([.9]), {}, 1) == 1.0
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement with scikit-learn for standard metrics, seqeval (`mode` default, IOB2) for entity F1, and a port of `eval.py`'s `_threshold` for recall-based ones. `load_custom` wraps the user function to the `(y, probs, params, pos)` signature.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: metrics registry`.

---

### Task 3: Data loading, checks, splits and vault

**Files:**
- Create: `src/finetune_kit/data.py`, `tests/test_data.py`, fixtures in `tests/conftest.py` (`seq_jsonl`, `tok_jsonl` writers)

**Interfaces:**
- Consumes: `finetune_kit.paths.vault_dir`
- Produces:
  - `load_rows(path: Path, task: str) -> list[dict]` — validates schema: seq-cls `{"id","text","label","group_id"?}`; token-cls `{"id","tokens","tags","group_id"?}` with `len(tokens)==len(tags)` and BIO/BIOES tags. Raises `DataError(msg)` naming file, line and problem. Duplicate ids raise.
  - `label_map(train_rows, task) -> list[str]` — sorted unique labels as strings (seq-cls) or tag set incl. `O` (token-cls). Label ints are stringified.
  - `check(exp_dir: Path, task: str, max_len: int, tokenizer, rows_by_split: dict[str, list[dict]]) -> dict` — returns report dict: `n_rows` per split, `label_dist`, `minority_share`, `token_len` `{p50,p95,max}`, `n_over_max_len`, `n_exact_dupes`, `n_cross_split_dupes` (by normalized text), `n_group_leak` (group_ids in >1 split), `unseen_labels` (in val/test not in train), `entity_token_share` (token-cls: share of non-`O` tags in train), `warnings: list[str]`, `errors: list[str]`.
  - `val_size_warning(task, objective: str, params: dict, val_label_counts: dict[str,int], positive: str|None) -> str|None` — recall-constrained objectives (`pauc_hi`, `skip_at_recall`, `precision_at_recall`): warn when `n_val_pos × (1 − recall_lo) < 10` (`recall_lo` = `recall_min` or `recall`); message states the positives needed and suggests a larger val share. `recall_at_precision`: warn when `n_val_pos < 100`. All other metrics: warn when the smallest val class (seq-cls) or least frequent entity type (token-cls) has < 100 examples. `check` appends the result to `warnings`.
  - `make_splits(rows, task, seed=0, frac=(0.8, 0.1, 0.1)) -> tuple[list, list, list]` — grouped by `group_id` when present (else by `id`), stratified by label (seq-cls) / by presence of any entity (token-cls).
  - `freeze_splits(exp_dir, uid, train, val, test) -> dict` — writes `splits/train.jsonl`, `splits/val.jsonl`, writes test to `vault_dir(uid)/test.jsonl` (chmod 600) and `splits/test.meta.json` `{"sha256","n_rows"}`; returns sha256s of all three.
  - `read_test(uid) -> list[dict]` (used only by promote).
  - `subset(rows, n, seed) -> list` — stratified subsample for research.

- [ ] **Step 1: Write failing tests:**

```python
def test_bad_json_line_names_line(tmp_path): ...  # DataError match "line 2"
def test_token_len_mismatch(tmp_path): ...       # DataError match "tokens/tags length"
def test_string_labels_mapped(tmp_path):          # labels "pii","clean" -> ["clean","pii"]
def test_unseen_label_reported(...):              # val has label "x" not in train -> report["unseen_labels"] == ["x"]
def test_group_split_no_leak():                   # 100 rows, 20 groups -> no group in 2 splits
def test_split_stratified():                      # 30% positives -> each split within ±5 points
def test_freeze_moves_test_to_vault(tmp_path):    # splits/test.jsonl absent; vault file mode 0o600; meta n_rows correct
def test_cross_split_dupe_counted():              # same text in train and val -> n_cross_split_dupes == 1
def test_over_max_len_counted():                  # tokenizer stub with len=600, max_len 512 -> n_over_max_len == 1
def test_val_warning_pauc():                      # pauc_hi recall_min .98, 266 val positives -> warning mentions "500"
def test_val_warning_none_when_enough():          # 600 positives -> None
def test_val_warning_small_class_f1():            # macro_f1, smallest class 40 -> warning
```
(Tokenizer stub: object with `__call__(texts, add_special_tokens=True)` returning `{"input_ids": [[0]*k ...]}`.)
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement as specified.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: data validation, checks, grouped splits and test vault`.

---

### Task 4: Model profiles and hyperparameter catalog

**Files:**
- Create: `src/finetune_kit/profiles.py`, `src/finetune_kit/profiles/models.yaml`, `src/finetune_kit/catalog.py`, `tests/test_catalog.py`

**Interfaces:**
- Produces:
  - `ModelProfile`: `key, hf_id, revision, tasks: list[str], lr_default: float, lr_range: tuple[float,float], max_len: int, batch: int, micro: int, force_fp32_on_mps: bool, grad_checkpointing: bool, trust_remote_code: bool, dropout_keys: dict[str, list[str]]` (`{"hidden": [...], "attn": [...]}` config attribute names), `lora_targets: list[str], notes: str`.
  - Dropout keys / LoRA targets (exact): BERT, RoBERTa: hidden `[hidden_dropout_prob]`, attn `[attention_probs_dropout_prob]`, LoRA `[query, key, value, dense]`; DeBERTa-v3: same dropout keys, LoRA `[query_proj, key_proj, value_proj, dense]`; ModernBERT: hidden `[embedding_dropout, mlp_dropout]`, attn `[attention_dropout]`, LoRA `[Wqkv, Wo, Wi]`; UltraBERT: read its config class at implementation time and fill in; generic profiles: use the BERT-style names if the config has them, else `{}` (backbone dropout and LoRA are then rejected by `validate_config` with a clear message).
  - `load_profiles(project_root: Path|None) -> dict[str, ModelProfile]` — shipped YAML, then `.finetune-kit/models.yaml` overrides/additions.
  - `add_profile(project_root, hf_id, trust_remote_code: bool) -> ModelProfile` — generic profile (lr 3e-5, range [1e-5, 1e-4], max_len from HF config `max_position_embeddings` capped 512 unless ModernBERT-style >512, batch 8, micro 4) with `notes: "generic: lr-scan required"`.
  - `catalog.ENTRIES: dict[str, Entry]`, `Entry(name, group, tasks, kind: Literal["float","int","choice","bool"], hard: tuple|list, log: bool, help: str)`.
  - `catalog.names(task) -> list[str]`; `catalog.listing(task, profile, report: dict, objective: str) -> list[dict]` (`name, group, default, help, recommended: bool, reason: str|None`).
  - `catalog.recommend(report, objective, task) -> dict[str, str]` (name → reason).
  - `catalog.validate_config(config: dict, tunable: list[str], losses: list[str], defaults: dict, task) -> list[str]` (empty = OK).
- Shipped profiles (exact):

| key | hf_id | revision | lr_default | lr_range | max_len | quirks |
|---|---|---|---|---|---|---|
| ultrabert-150m | dmis-lab/UltraBERT-150M | 545ac9d9c7aece444f014ada5a1e9818132b4d35 | 8e-5 | [2e-5, 1.5e-4] | 512 | trust_remote_code |
| deberta-v3-base | microsoft/deberta-v3-base | 8ccc9b6f36199bec6961081d44eb72fb3f7353f3 | 1e-5 | [5e-6, 3e-5] | 512 | force_fp32_on_mps, grad_checkpointing |
| deberta-v3-large | microsoft/deberta-v3-large | main→pin at implementation | 6e-6 | [3e-6, 1.5e-5] | 512 | force_fp32_on_mps, grad_checkpointing |
| roberta-base | FacebookAI/roberta-base | pin | 2e-5 | [1e-5, 5e-5] | 512 | |
| roberta-large | FacebookAI/roberta-large | pin | 1e-5 | [5e-6, 2e-5] | 512 | |
| modernbert-base | answerdotai/ModernBERT-base | pin | 5e-5 | [2e-5, 1e-4] | 512 (model allows 8192; default 512) | |
| modernbert-large | answerdotai/ModernBERT-large | pin | 3e-5 | [1e-5, 6e-5] | 512 | |
| bert-base | google-bert/bert-base-uncased | pin | 3e-5 | [1e-5, 5e-5] | 512 | |

  "pin" = the implementer looks up the current commit sha on the Hub once and writes it in. Batch 8, micro 4 for base models; batch 8, micro 2 for large.
- Catalog entries and hard limits (exact): `lr` [1e-7, 1e-3] log · `llrd` [0.5, 1.0] · `head_lr_mult` [1, 100] · `warmup` [0, 0.5] · `schedule` {linear, cosine, constant} · `weight_decay` [0, 0.5] · `epochs` [0.5, 20] · `batch` {4, 8, 16, 32, 64} · `clip` [0.1, 10] · `pooling` {cls, mean, attention, weighted_layers} (seq-cls) · `pool_layers` [2, 12] int (default 4; used by weighted_layers) · `msd_samples` [1, 8] int (default 1 = off; multi-sample dropout) · `hidden_dropout` [0, 0.3] (default = model's own) · `attn_dropout` [0, 0.3] (default = model's own) · `beta2` [0.95, 0.9999] (default 0.999) · `oversample` [1, 10] (default 1 = off; seq-cls; minority-class sampling factor) · `lora_r` {0, 4, 8, 16, 32, 64} (default 0 = off) · `lora_alpha` [1, 128] (default 16) · `head` {linear, mlp} · `dropout` [0, 0.6] · `loss` choice of allowed losses · `class_weight` [1, 20] or "balanced" · `focal_gamma` [0, 5] · `label_smoothing` [0, 0.3] · `rank_weight` [0, 10] (seq-cls) · `rdrop_alpha` [0, 10] · `ema` {0} ∪ [0.9, 0.9999] · `swa` bool · `reinit_top` [0, 6] int · `freeze_embed` bool · `fgm_eps` [0, 5] · `precision` {fp32, bf16} · `max_len` [32, profile.max_len] int · `crf` bool (token-cls) · `label_align` {first, all} (token-cls) · `asl_gamma_neg` [0, 8] (default 4) · `asl_gamma_pos` [0, 5] (default 0) · `asl_clip` [0, 0.2] (default 0.05) · `dice_smooth` [0.01, 10] log (default 1.0) · `dice_ce_weight` [0, 1] (default 0.5) (token-cls). Losses: seq-cls {ce, focal, asl, rank, ce+rank}; token-cls {ce, focal, dice}. Loss-specific keys (`focal_gamma`, `asl_*`, `rank_weight`, `dice_*`) are tunable only when that key is ticked in `/finetune-kit.hparams`; ticking a loss auto-ticks nothing.
- Recommendation rules (exact): `lr`, `llrd` always ("most sensitive" / "largest single gain in past tuning"); `minority_share < 0.3` → `class_weight`, `loss:focal`; objective in {skip_at_recall, precision_at_recall, pauc_hi} (recall-first) and seq-cls → `loss:asl` ("pushes harder on missed positives"); token-cls and `entity_token_share < 0.1` → `loss:dice` ("entities are rare"); objective in {pauc_hi, skip_at_recall, precision_at_recall, recall_at_precision, roc_auc, pr_auc} and seq-cls → `loss:ce+rank`; `token_len.p95 < 0.5 * max_len` → `max_len` ("shorter runs").
- `validate_config` rules: unknown key → error; non-numeric string for numeric key → error naming key; value differing from `defaults[key]` while key not in `tunable` → error; `loss` not in `losses` → error; outside hard limits → error; keys equal to default always OK.

- [ ] **Step 1: Write failing tests:**

```python
def test_profiles_load_and_pins():
    p = load_profiles(None)["ultrabert-150m"]; assert p.trust_remote_code and p.lr_default == 8e-5
def test_project_override(tmp_path)   # .finetune-kit/models.yaml changes roberta-base lr_default
def test_recommend_imbalance():
    r = recommend({"minority_share": 0.18, "token_len": {"p95": 400}}, "f1", "seq-cls")
    assert {"lr", "llrd", "class_weight", "loss:focal"} <= set(r)
def test_recommend_auc_metric_rank():  assert "loss:ce+rank" in recommend({...balanced...}, "pauc_hi", "seq-cls")
def test_recommend_asl_recall_first():  assert "loss:asl" in recommend({...balanced...}, "skip_at_recall", "seq-cls")
def test_recommend_dice_rare_entities(): assert "loss:dice" in recommend({"entity_token_share": 0.04, ...}, "entity_f1", "token-cls")
def test_dice_not_for_seq():  # {"loss": "dice"} for seq-cls -> error
def test_validate_locked_key():
    errs = validate_config({"warmup": 0.1}, ["lr"], ["ce"], DEFAULTS, "seq-cls")
    assert any("warmup" in e and "not tunable" in e for e in errs)
def test_validate_default_value_ok():   # {"warmup": 0.05} with warmup locked -> []
def test_validate_string_number():     # {"lr": "4e-5"} -> error mentions "lr" and "number"
def test_validate_hard_limit():        # {"lr": 0.5} -> error mentions "1e-3"
def test_validate_loss_not_allowed():  # {"loss": "rank"} with losses ["ce","focal"] -> error
def test_token_task_has_no_pooling():  assert "pooling" not in names("token-cls")
def test_backbone_dropout_rejected_without_keys():  # generic profile, dropout_keys {} -> {"hidden_dropout": 0.2} error mentions "not supported for this model"
def test_lora_alpha_requires_lora_r():  # {"lora_alpha": 32} with lora_r 0 -> error
```
- [ ] **Step 2–4:** Run → FAIL; implement; run → PASS.
- [ ] **Step 5:** Commit `feat: model profiles and hyperparameter catalog`.

---

### Task 5: Experiment files, state and checksums

**Files:**
- Create: `src/finetune_kit/experiment.py`, `tests/test_experiment.py`

**Interfaces:**
- Produces (pydantic v2 models, YAML on disk):
  - `ExperimentConfig`: `uid: str` (uuid4 hex), `name: str`, `task: Literal["seq-cls","token-cls"]`, `model: str`, `data: DataPaths(train: str, val: str|None, test: str|None)`, `objective: MetricRef(metric: str, params: dict = {})`, `guardrail: MetricRef|None`, `positive_class: str|None`, `custom_metric: CustomMetric(path: str, func: str, direction: str)|None`, `seeds: int = 3`, `budget: Budget(max_runs: int = 40, max_hours: float = 24)`, `research_subset: int|None`, `min_delta: float|None`, `versions: dict[str,str]`.
  - `SearchSpace`: `round: int`, `tunable: list[str]`, `losses: list[str]`.
  - `State`: `round: int = 0`, `checksums: dict[str,str]`, `sigma_obj: float|None`, `sigma_guard: float|None`, `noise_run: str|None`, `best_run: str|None`, `lr_scan_best: float|None`, `baseline_train_s: float|None`, `non_keep_streak: int = 0`, `runs_used: int = 0`, `hours_used: float = 0`, `promoted: bool = False`, `bakeoff: dict|None` (last `bakeoff_rank` output), `model_locked: bool = False` (set by `bakeoff choose` or the first noise run; `finetune-kit bakeoff` refuses after).
  - `new_experiment(root: Path, name: str, cfg: ExperimentConfig) -> Path` — creates `experiments/NNN-<slug>/` (next number, zero-padded 3), writes `experiment.yaml`, `state.json`.
  - `load(exp_dir) -> tuple[ExperimentConfig, SearchSpace|None, State]`; `save_state(exp_dir, State)`.
  - `commit_space(exp_dir, space: SearchSpace) -> State` — validates names against the catalog (Task 4 `catalog.names(task)`), increments `round`, writes `search-space.yaml`, updates `checksums["search-space.yaml"]`.
  - `protected_files(exp_dir, project_root) -> list[Path]` = `.finetune-kit/memory/research-rules.md`, `search-space.yaml`, `splits/train.jsonl`, `splits/val.jsonl`, `splits/test.meta.json`.
  - `record_checksums(exp_dir, project_root) -> State`; `verify_checksums(exp_dir, project_root) -> None` (raises `TamperError` listing changed files).

- [ ] **Step 1: Write failing tests:**

```python
def test_new_experiment_numbering(tmp_path):  # second call -> "002-..."
def test_roundtrip_yaml(tmp_path):            # save/load equal
def test_commit_space_increments_round(tmp_path)
def test_commit_space_rejects_unknown(tmp_path):  # tunable ["nope"] -> ValueError match "nope"
def test_tamper_detected(tmp_path):           # record, edit search-space.yaml by hand, verify -> TamperError match "search-space.yaml"
def test_commit_after_record_is_not_tamper(tmp_path)
```
- [ ] **Step 2–4:** Run → FAIL; implement; run → PASS.
- [ ] **Step 5:** Commit `feat: experiment config, state and tamper checks`.

---

### Task 6: Keep/discard statistics

**Files:**
- Create: `src/finetune_kit/stats.py`, `tests/test_stats.py`

**Interfaces:**
- Produces: `@dataclass class RunSummary: objective_mean: float; objective_std: float; guardrail_mean: float|None; n_seeds: int; n_nondefault: int`; `verdict(cand: RunSummary, best: RunSummary, sigma_obj: float, sigma_guard: float|None, direction: Literal["max","min"], min_delta: float, guard_direction: Literal["max","min"] = "max") -> Literal["keep","keep-simpler","discard"]`; `sigma(values: list[float]) -> float` (sample std, ddof=1; 0.0 for <2 values).
  - `bakeoff_rank(results: dict[str, tuple[list[float], float]], direction, min_delta) -> dict` — input model → (per-seed objectives, mean train_s); pooled σ = sqrt(mean of per-model variances); N = min seeds; leader = best mean; `clear = Δ(leader, runner_up) > max(2·σ_pooled·sqrt(2/N), min_delta)`; returns `{"ranking": [{model, mean, std, train_s}...], "winner": leader if clear else the fastest model whose mean is within the bar of the leader, "clear": bool, "reason": str}`.

- [ ] **Step 1: Write failing tests** (σ = 0.03, N = 3 → noise = 0.02449, bar = 0.04899):

```python
B = RunSummary(0.70, 0.01, 0.40, 3, 0)
def S(obj, guard=0.40, nd=1): return RunSummary(obj, 0.01, guard, 3, nd)

def test_keep():            assert verdict(S(0.75), B, 0.03, 0.02, "max", 0.002) == "keep"
def test_below_bar():       assert verdict(S(0.745), B, 0.03, 0.02, "max", 0.002) == "discard"
def test_min_delta_wins():  assert verdict(S(0.7015), B, 0.0, None, "max", 0.002) == "discard"
def test_guardrail_drop():  assert verdict(S(0.80, guard=0.37), B, 0.03, 0.02, "max", 0.002) == "discard"  # drop .03 > .02*sqrt(2/3)=.0163
def test_guardrail_small_drop_ok(): assert verdict(S(0.80, guard=0.39), B, 0.03, 0.02, "max", 0.002) == "keep"
def test_simpler():
    best = RunSummary(0.75, 0.01, 0.40, 3, 3)
    assert verdict(S(0.74, nd=1), best, 0.03, 0.02, "max", 0.002) == "keep-simpler"
def test_min_direction():   assert verdict(S(0.30), RunSummary(0.40, .01, None, 3, 0), 0.03, None, "min", 0.005) == "keep"
def test_sigma():           assert sigma([0.7241, 0.6682, 0.6698]) == pytest.approx(0.0318, abs=1e-4)
def test_bakeoff_clear_winner():     # a:[.80,.81,.79] 2000s, b:[.70,.71,.69] 1000s -> winner a, clear True
def test_bakeoff_tie_picks_faster():  # a:[.72,.66,.67] 2000s, b:[.70,.64,.69] 1000s -> clear False, winner b
```
- [ ] **Step 2–4:** Run → FAIL; implement per Global Constraints keep rule (Δ = cand − best for max, best − cand for min); run → PASS.
- [ ] **Step 5:** Commit `feat: noise-aware keep/discard verdict`.

---

### Task 7: Trainer building blocks (config, optimizer, losses, models)

**Files:**
- Create: `src/finetune_kit/train/__init__.py`, `train/config.py`, `train/optim.py`, `train/losses.py`, `train/models.py`, `tests/test_train_parts.py`

**Interfaces:**
- Consumes: `ModelProfile`, `catalog.ENTRIES`
- Produces:
  - `TrainConfig` (pydantic, `extra="forbid"`): every catalog key, plus `task`, `model`, `seed: int`, `micro: int`, `beta1: float = 0.9` (fixed), `eps: float = 1e-6`, `eval_every: float = 0.5`, `max_steps: int|None`, `step_fraction: float = 1.0` (LR scan). Defaults = `sweep_train.py` DEFAULTS (lr from profile, epochs 2, batch 8, warmup 0.05, linear, wd 0.01, clip 1.0, dropout 0.1, pooling cls, head linear, llrd 1.0, head_lr_mult 1.0, loss ce, class_weight 1.0, focal_gamma 2.0, label_smoothing 0, rank_weight 1.0, ema 0, reinit_top 0, freeze_embed false, precision fp32) plus `rdrop_alpha 0, fgm_eps 0, swa false, crf false, label_align first, max_len profile.max_len, pool_layers 4, msd_samples 1, hidden_dropout None, attn_dropout None, beta2 0.999, oversample 1, lora_r 0, lora_alpha 16`.
  - `defaults_for(profile, task) -> dict`; `build_config(overrides: dict, profile, task, seed) -> TrainConfig`; `nondefault_count(overrides, defaults) -> int`.
  - `param_groups(model, cfg) -> list[dict]` — port of `sweep_train.param_groups` (head params named `head.*` / `crf.*` get `head_lr_mult`; layer index regex `\.layers?\.(\d+)\.`; embeddings depth 0; no decay for ndim < 2); each group has `base_mult`.
  - `lr_factor(step, steps, cfg) -> float` — port of `sweep_train.lr_factor`.
  - `class_weights(cfg, train_label_ids, n_classes, pos) -> torch.Tensor|None` (float → weight on class `pos`, others 1; "balanced" → n/(C·count_c); 1.0 → None).
  - `seq_loss(logits, y, cfg, weights: torch.Tensor|None, pos: int) -> torch.Tensor` (sum over batch); rank terms use the `logits[:, pos]` margin and are allowed only when C == 2.
  - `token_loss(logits, labels(-100 ignored), cfg, weights, o_id: int) -> torch.Tensor`. `dice`: `dice_ce_weight · CE + (1 − dice_ce_weight) · (1 − soft Dice)`, soft Dice = mean over non-`O` tag classes of `(2·Σp·y + s)/(Σp + Σy + s)` with `s = dice_smooth`, over non-ignored tokens of the batch. Not allowed with `crf: true` (validate_config error).
  - `asl` (seq-cls, in `seq_loss`): one-vs-rest sigmoid on logits; positives `−(1−p)^γ+ · log p`, negatives with shifted prob `p_m = max(p − asl_clip, 0)`: `−p_m^γ− · log(1 − p_m)`; summed over classes and batch (Ridnik et al., 2021). Probabilities for metrics stay softmax.
  - `SeqClassifier(backbone, n_classes, cfg)` (pooling cls/mean/attention/weighted_layers — the CLS vectors of the last `pool_layers` hidden states mixed by softmax-normalised learnable weights, needs `output_hidden_states=True`; head linear/mlp; with `msd_samples > 1` the head runs `msd_samples` times with independent dropout and logits are averaged in training, single pass at eval); `TokenClassifier(backbone, n_tags, cfg)` with optional `torchcrf.CRF`; both `forward(input_ids, attention_mask) -> logits`.
  - `encode_seq(tok, texts, max_len) -> (list[list[int]], n_truncated)`; `encode_tokens(tok, tokens, tags, tag2id, max_len, align) -> (ids, label_ids, word_ids, n_truncated)` (fast tokenizer `is_split_into_words=True`; `first`: label first sub-token, others −100; `all`: all sub-tokens get the word's tag with B-→I- on non-first); `collate(...)` pads.
  - `load_backbone(profile, device, hidden_dropout: float|None, attn_dropout: float|None) -> nn.Module` (sets every config key in `profile.dropout_keys` before loading; fp32 when `force_fp32_on_mps` and device mps; `revision` and `trust_remote_code` from profile).
  - `apply_lora(backbone, profile, r, alpha) -> nn.Module` (peft `LoraConfig(r, lora_alpha=alpha, target_modules=profile.lora_targets, lora_dropout=0.05)`; head stays fully trainable; `param_groups` applies LLRD to LoRA params by their layer index).

- [ ] **Step 1: Write failing tests:**

```python
def test_llrd_multipliers():     # stub model with config.num_hidden_layers=2; llrd=0.5
    mults = {g["base_mult"] for g in param_groups(stub, cfg)}; assert {1.0, 0.5, 0.25} <= mults
def test_lr_factor_warmup_and_end(): # steps=100 warmup .05 -> f(0)=0.2, f(4)=1.0, f(99)<0.02; cosine midpoint≈0.5
def test_focal_gamma0_equals_ce()
def test_asl_zero_gammas_zero_clip_equals_bce()   # asl_gamma_neg=0, asl_gamma_pos=0, asl_clip=0 == BCE-with-logits on one-hot, summed
def test_asl_clip_ignores_easy_negatives()       # negative with p=0.03 and clip 0.05 -> zero loss contribution
def test_dice_weight1_equals_ce()                 # dice_ce_weight=1 -> equals token CE
def test_dice_perfect_prediction_near_zero()      # one-hot-ish logits (±20) matching labels, dice_ce_weight=0 -> loss < 1e-3
def test_dice_ignores_padding()                   # labels -100 positions do not change loss
def test_ce_rank_weight0_equals_ce()
def test_class_weight_balanced_inverse_freq()
def test_unknown_key_rejected():   with pytest.raises(ValidationError): build_config({"nope": 1}, prof, "seq-cls", 0)
def test_mean_pooling_ignores_padding()  # same output with extra padded positions
def test_weighted_layers_pooling()      # logits (B, C); layer-weight parameter has size pool_layers
def test_msd_eval_is_deterministic()   # model.eval(): two forwards equal
def test_backbone_dropout_applied()    # tiny-bert hidden_dropout .25 -> backbone.config.hidden_dropout_prob == .25
def test_lora_trainable_params_small() # lora_r 8 -> trainable backbone params < 5% of backbone total; head params trainable
def test_token_truncation_keeps_alignment():  # Review Focus 5
    ids, labels, words, n_trunc = encode_tokens(tok, [["John"]+["x"]*600], [["B-PER"]+["O"]*600], t2i, 64, "first")
    assert n_trunc == 1 and len(ids[0]) == 64 and labels[0][1] == t2i["B-PER"] and labels[0][0] == -100 and labels[0][-1] == -100
def test_align_all_converts_B_to_I()
```
Use `hf-internal-testing/tiny-random-bert` tokenizer for token tests (`@pytest.mark.model`).
- [ ] **Step 2–4:** Run → FAIL; implement; run → PASS.
- [ ] **Step 5:** Commit `feat: trainer config, optimizer, losses and heads`.

---

### Task 8: Training loop

**Files:**
- Create: `src/finetune_kit/train/loop.py`, `src/finetune_kit/train/worker.py`, `tests/test_loop.py`

**Interfaces:**
- Consumes: Task 7 everything; `metrics.compute`
- Produces:
  - `@dataclass TrainResult: val_out: np.ndarray | list[list[str]]` (probs (n,C) for seq-cls; predicted tag lists for token-cls), `best_epoch: float, train_s: float, peak_gb: float, n_truncated: int, steps: int, model: nn.Module`
  - `train_and_predict(cfg: TrainConfig, profile, train_rows, val_rows, labels: list[str], select: Callable[[np.ndarray|list], float], device: str, keep_model: bool = False) -> TrainResult` — steps = `max_steps or round(epochs * len(train) / batch * step_fraction)`; micro-batch accumulation; grad clip; env `FINETUNE_KIT_MAX_STEPS` (tests only) caps steps; EMA via `torch._foreach_lerp_` (predict with EMA weights); SWA averages weights from the last 25% of steps; FGM perturbs word embeddings by `fgm_eps` on a second forward/backward; R-Drop adds `rdrop_alpha * symmetric KL` of two dropout passes; AdamW betas `(0.9, beta2)`; `oversample > 1` (seq-cls) builds each epoch's order with minority-class rows repeated `round(oversample)` times, shuffled with the seed (`steps` uses the oversampled epoch length); bf16 autocast when `precision == "bf16"`; evaluates val every `eval_every` epochs and at the end, keeps the best state by `select` (direction-aware via the metric) in memory; raises `RuntimeError("non-finite loss at step N")`.
  - `python -m finetune_kit.train.worker --exp DIR --run RUN_ID --seed K` — loads experiment, config from `runs/<run>/request.json`, rows from splits (subset when `research_subset` and stage ≠ promote), trains, computes objective, guardrail and the `report` metrics on val, writes `runs/<run>/seed_<k>/metrics.json` `{objective, guardrail, report: {...}, train_s, peak_gb, n_truncated, best_epoch, device}` and `val_preds.npy`/`.json`. Exit code 0 on success, 1 on error with message in `seed_<k>/error.txt`.

- [ ] **Step 1: Write failing tests** (`@pytest.mark.model`, tiny-random-bert, CPU, `max_steps=4`):

```python
def test_seq_binary_runs_and_scores(seq_rows):    # probs shape (n_val, 2), rows sum to 1, train_s > 0
def test_seq_multiclass_runs(seq3_rows)
def test_token_runs_with_crf(tok_rows)             # val_out is list of tag lists same lengths as inputs
def test_ema_swa_fgm_rdrop_bf16_smoke(seq_rows)    # each flag individually: no exception, finite probs
def test_new_tunables_smoke(seq_rows)              # weighted_layers, msd_samples 4, hidden_dropout .2, beta2 .98, oversample 3, lora_r 8: each runs
def test_oversample_epoch_composition()            # 10 pos / 90 neg, oversample 3 -> one epoch order has 30 pos, 90 neg
def test_same_seed_same_scores(seq_rows)           # two runs seed 0 -> allclose(atol=1e-5) on CPU
def test_worker_writes_metrics(tmp_exp)            # metrics.json has objective, guardrail keys
```
- [ ] **Step 2–4:** Run `uv run pytest tests/test_loop.py -m model -v` → FAIL; implement; → PASS.
- [ ] **Step 5:** Commit `feat: training loop and per-seed worker`.

---

### Task 9: Run requests, queue and worker daemon

**Files:**
- Create: `src/finetune_kit/store.py`, `src/finetune_kit/queue.py`, `tests/test_queue.py`
- Modify: `src/finetune_kit/cli.py` (add `run`, `wait`, `status`, `lr-scan`, `_worker` hidden)

**Interfaces:**
- Consumes: `experiment.load/verify_checksums/save_state`, `catalog.validate_config`, `stats.verdict/sigma`, `train.config.build_config/nondefault_count`
- Produces:
  - `store.new_run(exp_dir, stage: Literal["bakeoff","lr-scan","noise","research","promote"], overrides: dict, seeds: list[int], description: str) -> str` (run id `r<NNN>`, writes `runs/<id>/request.json` with `status: "queued"`, `created`).
  - `store.summarize(exp_dir, run_id) -> RunSummary`; `store.append_result(exp_dir, row: dict)` — `results.tsv` columns exactly: `run_id, round, stage, objective_mean, objective_std, guardrail_mean, n_seeds, minutes, verdict, description, config`.
  - `queue.enqueue(exp_dir, run_id)` — appends to `kit_home()/queue.jsonl`, starts daemon (`subprocess.Popen([sys.executable, "-m", "finetune_kit.cli", "_worker"], start_new_session=True)`) unless `kit_home()/worker.lock` holds a live PID.
  - `queue.worker_loop()` — on start, `recover_stale()`; FIFO; per seed runs `python -m finetune_kit.train.worker` with timeout (`3 × state.baseline_train_s` for research/promote; 4 h for bakeoff, lr-scan and noise); updates `request.json` status (`running` with pid → `done`|`crash`|`timeout`); after all seeds: noise stage sets `state.sigma_obj/sigma_guard/noise_run/best_run/baseline_train_s`; lr-scan stage updates `state.lr_scan_best` when all scan runs are done; research stage computes verdict vs `state.best_run`, updates `best_run` on keep/keep-simpler, `non_keep_streak`, `runs_used`, `hours_used`; appends results row. Exits when queue empty.
  - `queue.recover_stale()` — `running` request with dead pid → `crash`; re-enqueue once if `retries == 0` (increments `retries`).
- CLI:
  - `finetune-kit run EXP --config JSON [--seeds N] [--stage research|noise] [--desc TEXT]` → checks: `verify_checksums`; research requires `state.sigma_obj is not None` and search space present; budget (`runs_used < max_runs`, `hours_used < max_hours`); version check (torch/transformers/finetune-kit/profile revision differ from `experiment.yaml` → `"warnings"` list in output, not an error); `validate_config` errors → `fail`; emits `{"run_id", "status": "queued", "position"}`. Noise stage forces default config (+ `lr = state.lr_scan_best` if set) and records checksums on first call.
  - `finetune-kit bakeoff EXP --models a,b[,c]` → refuses if `state.model_locked`; each model must support the task; enqueues one run per model (stage bakeoff, profile defaults incl. profile LR, `cfg.seeds` seeds, `overrides["model"] = key`); when all are done the worker stores `stats.bakeoff_rank(...)` in `state.bakeoff`. Without `--models` it returns `state.bakeoff`.
  - `finetune-kit bakeoff choose EXP MODEL` → sets `model` in `experiment.yaml` (and the profile revision in `versions`), `state.model_locked = True`.
  - `finetune-kit lr-scan EXP` → enqueues 5 runs, LRs log-spaced over `profile.lr_range`, `step_fraction 0.2`, seed 0, stage lr-scan; emits run ids.
  - `finetune-kit wait RUN [--timeout 540]` → blocks until done or timeout; emits `{"run_id","status","seeds":[...metrics], "verdict"?}`; on timeout `{"status":"running","elapsed_s"}` (exit 0).
  - `finetune-kit status [EXP]` → queue and running job.

- [ ] **Step 1: Write failing tests** (use a fake worker: env `FINETUNE_KIT_FAKE_TRAIN=1` makes `finetune_kit.train.worker` write `metrics.json` with objective from `request.json` `"_fake": [..per seed..]` and sleep 0):

```python
def test_research_requires_noise_floor(exp)        # finetune-kit run --stage research before noise -> error "noise floor"
def test_locked_key_rejected(exp_with_noise)        # -> exit 2, error mentions "warmup"
def test_string_number_rejected(exp_with_noise)     # Review Focus 2
def test_noise_sets_sigma(exp)                      # fake [0.7241,0.6682,0.6698] -> state.sigma_obj≈0.0318
def test_keep_updates_best(exp_with_noise)          # fake mean +0.06 -> verdict keep, state.best_run == run
def test_streak_counts(exp_with_noise)              # 2 discards -> non_keep_streak == 2
def test_budget_exhausted(exp_with_noise)           # max_runs=1 used -> error "budget"
def test_tamper_blocks_run(exp_with_noise)          # edit search-space.yaml -> error "changed"
def test_stale_running_recovered(exp)               # Review Focus 3: request status running, pid 999999 -> crash, retried once
def test_timeout_marks_timeout(exp_with_noise)      # fake sleeps beyond timeout (set baseline_train_s tiny) -> status timeout, verdict discard
def test_results_tsv_columns(exp_with_noise)
def test_bakeoff_records_ranking(exp)              # fake objectives for 2 models -> state.bakeoff["winner"] set
def test_bakeoff_refused_after_lock(exp)           # after choose -> error "model is locked"
def test_noise_locks_model(exp)                     # first noise run sets model_locked
```
- [ ] **Step 2–4:** Run → FAIL; implement; → PASS.
- [ ] **Step 5:** Commit `feat: run queue, detached worker and verdict bookkeeping`.

---

### Task 10: Information, data and comparison commands

**Files:**
- Create: `src/finetune_kit/report.py`, `tests/test_cli_commands.py`
- Modify: `src/finetune_kit/cli.py` (add `new`, `models`, `metrics`, `check-data`, `catalog`, `space commit`, `compare`, `report`, `models add`)

**Interfaces:**
- Consumes: Tasks 2–6, 9
- Produces CLI (all JSON):
  - `finetune-kit new NAME [--custom-metric PATH:FUNC --direction max|min] --task T --model M --train PATH [--val PATH] [--test PATH] --metric NAME [--metric-params JSON] [--guardrail NAME] [--guardrail-params JSON] [--positive-class L] [--seeds 3] [--max-runs 40] [--max-hours 24] [--subset N]` → creates experiment, records versions (`finetune-kit`, `torch`, `transformers`, profile revision); emits `{"experiment": "001-name", "path"}`. Rejects metric not valid for task, and `needs_positive` metric on multi-class without `--positive-class`.
  - `finetune-kit models [--task]`, `finetune-kit models add HF_ID [--trust-remote-code]`, `finetune-kit metrics [--task]`.
  - `finetune-kit check-data EXP` → runs `data.check`; if no val/test given, splits and freezes; else freezes given files (test → vault); writes `data-report.json`; emits report. Refuses to re-split once `splits/` exists unless `--resplit` (which also resets state); `--val-frac F` (default 0.1, test share stays 0.1) sets the validation share when splitting.
  - `finetune-kit catalog EXP` → `catalog.listing`.
  - `finetune-kit space commit EXP --tunable a,b,c [--losses ce,focal]` → `experiment.commit_space`.
  - `finetune-kit misses EXP RUN [--k 20]` → best seed's val predictions vs val rows. seq-cls: `{"false_negatives": [{id, text (first 300 chars), label, score}...], "false_positives": [...], "possible_label_issues": [...]}` — for constrained objectives misses are judged at the val threshold from `metrics.threshold_for`, otherwise argmax; sorted by how wrong; `possible_label_issues` = rows scored > 0.95 against their label. token-cls: up to k rows with entity errors, each `{id, text, gold: [(type, start, end, text)], pred: [...]}`.
  - `finetune-kit compare EXP` → `{"best": run_id, "noise": {...}, "streak": n, "stop": bool, "budget_left": {...}, "runs": [rows...]}`; `stop` true when streak ≥ 10 or budget spent.
  - `finetune-kit report EXP` → writes `report.html` (self-contained: table of runs, objective per run as inline SVG dot plot with ±std bars and the keep bar, verdicts coloured) and prints its path.

- [ ] **Step 1: Write failing tests:** `test_new_rejects_bad_metric_for_task`, `test_new_requires_positive_class_multiclass`, `test_check_data_splits_and_vaults`, `test_check_data_refuses_resplit`, `test_catalog_marks_recommended`, `test_space_commit_cli`, `test_compare_stop_flag`, `test_report_writes_html` (file exists, contains every run id), `test_misses_seq_sorted` (fake preds: false negatives ascending by score, k respected), `test_misses_label_issue_flagged` (score .98 on a negative → in `possible_label_issues`), `test_misses_token_spans`, `test_check_data_reports_val_warning`.
- [ ] **Step 2–4:** Run → FAIL; implement; → PASS.
- [ ] **Step 5:** Commit `feat: setup, data, catalog, compare and report commands`.

---

### Task 11: Promote

**Files:**
- Create: `src/finetune_kit/promote.py`, `tests/test_promote.py`
- Modify: `src/finetune_kit/cli.py` (add `promote`)

**Interfaces:**
- Consumes: Tasks 2, 3, 8, 9
- Produces: `promote(exp_dir, run_id, seeds: int = 3) -> dict` and `finetune-kit promote EXP RUN [--seeds 3]`.
  - Refuses if `state.promoted` (error "already promoted; test is scored once") or run verdict not in {keep, keep-simpler} and not the noise run.
  - Retrains the run's config on the **full** train split (no subset) for `seeds` seeds (synchronously in the worker queue, stage `promote`), picks the seed with the best val objective, computes threshold on val via `metrics.threshold_for` when the objective (or guardrail) is constrained, reads test via `data.read_test(uid)`, scores test once (objective, guardrail, report metrics, and at threshold: recall, precision, skip/accuracy).
  - Writes `final/model/` (`save_pretrained` backbone + head state dict + tokenizer), `final/recipe.json` (full TrainConfig + profile + versions), `final/threshold.json` (if any), `final/test_metrics.json`, `final/val_metrics.json`; sets `state.promoted = True`.
  - Throughput: `benchmark(model, tok, rows, device) -> dict` on up to 200 val rows at batch 8 after 1 warm-up batch: `rows_per_s` on the training device and on CPU, `p50_latency_ms` per batch, `model_mb` (size of `final/model/` on disk); written to `final/throughput.json` and included in the promote output.
  - Export: `final/predict.py` rendered from `templates/predict.py.tmpl` — standalone (torch + transformers + peft only, no `finetune-kit` import): `python predict.py --input rows.jsonl --output scores.jsonl [--device cpu]`; reads `text` (seq-cls) or `tokens` (token-cls), writes `{id, score|label}` (seq-cls: positive-class score and `flag` when a threshold exists) or `{id, tags}`; loads `model/`, `recipe.json`, `threshold.json` from its own folder.

- [ ] **Step 1: Write failing tests** (`@pytest.mark.model`, tiny model, max_steps small): `test_promote_writes_final`, `test_promote_twice_refused`, `test_promote_discarded_run_refused`, `test_threshold_saved_for_recall_metric`, `test_no_threshold_for_f1`, `test_throughput_written` (keys rows_per_s, model_mb > 0), `test_exported_predict_matches` (run `final/predict.py` in a subprocess on 3 val rows; scores equal promote's val scores, atol 1e-5).
- [ ] **Step 2–4:** Run → FAIL; implement; → PASS.
- [ ] **Step 5:** Commit `feat: promote with full retrain and one-time test scoring`.

---

### Task 12: Installer, agent commands, rules and guard hook

**Files:**
- Create: `src/finetune_kit/installer.py`, `src/finetune_kit/guard.py`, `src/finetune_kit/templates/commands/*.md` (8), `src/finetune_kit/templates/predict.py.tmpl` (rendered in Task 11; created here only if Task 11 has not), `src/finetune_kit/templates/memory/research-rules.md`, `src/finetune_kit/templates/docs/*.md` (4), `tests/test_installer.py`
- Modify: `src/finetune_kit/cli.py` (add `init`, hidden `_guard`)

**Interfaces:**
- Produces: `finetune-kit init --ai claude|codex|cursor [--here] [--force]`; `installer.install(root: Path, ai: str, force: bool) -> list[Path]`; `guard.main()` reads Claude Code PreToolUse JSON on stdin, exits 2 with a message when `tool_input.file_path` (or `path`) resolves inside `.finetune-kit/memory/`, `experiments/*/splits/`, `experiments/*/final/`, `experiments/*/runs/`, or is `experiments/*/search-space.yaml`, `state.json`, `results.tsv`; else exit 0.
- Output locations (exact):
  - all: `.finetune-kit/memory/research-rules.md`, `.finetune-kit/templates/*.md`, `experiments/` (empty, with `.gitkeep`).
  - claude: `.claude/skills/finetune-kit-<stage>/SKILL.md` (frontmatter `name: finetune-kit-<stage>`, `description:` one line); hook merged into `.claude/settings.json` → `hooks.PreToolUse[] = {"matcher": "Edit|Write|MultiEdit|NotebookEdit", "hooks": [{"type": "command", "command": "finetune-kit _guard"}]}` (append if absent; keep existing entries).
  - codex: `.agents/skills/finetune-kit-<stage>/SKILL.md` (same frontmatter) and an `AGENTS.md` section between `<!-- finetune-kit:start -->` / `<!-- finetune-kit:end -->` markers pointing at the rules.
  - cursor: `.cursor/commands/finetune-kit.<stage>.md` and `.cursor/rules/finetune-kit.mdc` (always-apply) with the same pointer.
  Verify these paths against current spec-kit/agent docs at implementation time; if one differs, change only the path constant in `installer.AGENTS`.
- Template content requirements (canonical body shared by all agents; `{{CMD}}` is replaced with the agent's invocation prefix, e.g. `/finetune-kit-` or `/finetune-kit.`):
  - Every command: "Ask one question at a time"; read `.finetune-kit/memory/research-rules.md` first; refuse and name the missing stage if the previous stage's file is absent; call only `finetune-kit …` commands for actions; never open files under `~/.finetune-kit/` or `experiments/*/splits/`.
  - `setup`: asks, in order, task (list both, one line each) → dataset path(s) → model (from `finetune-kit models --task`, show lr/notes; offer "bake-off": pick 2–3 candidates, the first is used as a placeholder until `/finetune-kit-bakeoff`) → objective metric (from `finetune-kit metrics --task`, one line each; ask params when the metric has them; ask positive class when needed) → optional guardrail → seeds (default 3) → budget → research subset (offer when train > 5,000 rows) → runs `finetune-kit new` → writes `experiment.md` from template summarizing choices and why.
  - `check`: runs `finetune-kit check-data`; explains each finding in plain words; for errors, proposes a fix and stops; writes `data-report.md`.
  - `bakeoff`: only when setup chose bake-off (otherwise says it's skipped and points to hparams); runs `finetune-kit bakeoff --models`, polls each run with `finetune-kit wait`, shows the ranking with mean ± std and train time, explains whether the winner is clear or a tie (tie → faster model recommended), asks the user to confirm, runs `finetune-kit bakeoff choose`; writes `bakeoff.md`.
  - `check` also: reads `warnings` and, for the small-validation warning, explains why (noise) and offers to re-split with a bigger val share via `finetune-kit check-data --resplit --val-frac X`.
  - `hparams`: runs `finetune-kit catalog`; presents groups with ★ and reasons; asks which may be tuned (names only); confirms the list; runs `finetune-kit space commit`; writes `search-space.md`. On rerun, states it starts round N+1 and that best and σ carry over.
  - `baseline`: if `lr` tunable → `finetune-kit lr-scan`, `finetune-kit wait` each, report best LR; then `finetune-kit run --stage noise`, wait (poll `finetune-kit wait --timeout 540` in a loop), report σ; writes `baseline.md`.
  - `research`: loop — read `finetune-kit compare`; if `stop`, write `summary` section in `log.md` and end; propose ONE change with a one-line hypothesis appended to `log.md`; `finetune-kit run … --desc`; poll `finetune-kit wait`; record the verdict and what it taught in `log.md`; next. Resuming: start by reading `log.md` and `finetune-kit compare`. Never ask the human whether to continue.
  - `analyze`: `finetune-kit compare` + `finetune-kit report` + `finetune-kit misses <best>`; write `report.md`: best vs baseline with ± std, what helped/hurt, untried tunables to open next, and an **error analysis** section: patterns in the misses and false alarms, and each `possible_label_issues` row listed as a likely labelling problem for the user to check (never edits data).
  - `promote`: confirm run choice with the user, warn test is scored once, `finetune-kit promote`, summarize `final/test_metrics.json`, threshold and throughput (with an hourly estimate), and show how to run `final/predict.py`.
  - `research-rules.md`: rules 1–6 from spec §7 verbatim in meaning, plus the keep formula and stop rule from Global Constraints.

- [ ] **Step 1: Write failing tests:**

```python
def test_init_claude_writes_skills_and_hook(tmp_path)   # 8 SKILL.md files; settings.json has finetune-kit _guard hook
def test_init_merges_existing_settings(tmp_path)        # Review Focus 4: pre-existing hook kept, finetune-kit hook added once (run init twice)
def test_init_keeps_user_files_and_experiments(tmp_path)
def test_init_codex_and_cursor_paths(tmp_path)
def test_agents_md_section_replaced_not_duplicated(tmp_path)
def test_guard_blocks_splits(tmp_path)                  # stdin JSON for Edit on experiments/001-x/splits/train.jsonl -> exit 2
def test_guard_allows_log_md(tmp_path)                  # experiments/001-x/log.md -> exit 0
def test_templates_have_no_unfilled_placeholders(tmp_path)  # no "{{" in written files
```
- [ ] **Step 2–4:** Run → FAIL; implement installer + guard + write the 13 template files; → PASS.
- [ ] **Step 5:** Commit `feat: finetune-kit init for Claude Code, Codex and Cursor with guard hook`.

---

### Task 13: End-to-end smoke test and README

**Files:**
- Create: `tests/test_e2e.py`, `README.md`

**Interfaces:**
- Consumes: the whole CLI.

- [ ] **Step 1: Write the e2e test** (`@pytest.mark.model`, tiny-random-bert, CPU, env `FINETUNE_KIT_MAX_STEPS=4` honoured by the worker for tests only, seeds 2):
  `finetune-kit init --ai claude` → `finetune-kit new e2e --task seq-cls --model tiny-bert --train data.jsonl --metric pauc_hi` → `finetune-kit check-data` → `finetune-kit bakeoff --models tiny-bert,tiny-roberta` + waits → `finetune-kit bakeoff choose e2e tiny-bert` → `finetune-kit space commit --tunable lr,llrd --losses ce` → `finetune-kit lr-scan` + waits → `finetune-kit run --stage noise` + wait → `finetune-kit run --config '{"llrd":0.8}'` + wait → `finetune-kit compare` (2 runs, verdict present) → `finetune-kit report` (html exists) → `finetune-kit misses <best>` (JSON has false_negatives) → `finetune-kit promote <best>` (final/ files exist, test_metrics has pauc_hi, throughput.json exists, `python final/predict.py` runs). Repeat a shorter path for `token-cls` with `entity_f1`.
- [ ] **Step 2:** Run `uv run pytest -m model tests/test_e2e.py -v` → fix until PASS.
- [ ] **Step 3:** Run full suite `uv run pytest -v` → all PASS.
- [ ] **Step 4:** Write `README.md`: install (`uv tool install git+<repo>`), `finetune-kit init --ai …`, the eight commands, data formats, where things are stored, the keep rule in one paragraph.
- [ ] **Step 5:** Manual check: `uv tool install .` then in a temp project `finetune-kit init --ai claude`, open Claude Code, run `/finetune-kit-setup` against `tests/fixtures` data to confirm the conversation flows. Record anything odd in README "Known issues".
- [ ] **Step 6:** Commit `test: end-to-end smoke; docs: README`.
