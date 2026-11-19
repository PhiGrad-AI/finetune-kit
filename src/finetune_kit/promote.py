"""Promote: final retrain on the full train split, one-time test scoring, throughput and export.

`promote()` validates and queues a `stage=promote` run like every other run. The worker trains
its seeds on the full train split and saves each seed's model; `queue._finalize` then calls
`finish()`, which picks the best seed on val, sets the threshold on val, scores the test split
once and writes `final/`. This module is the only caller of `data.read_test`.
"""
import json
import os
import shutil
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

from finetune_kit import data, store
from finetune_kit import experiment as ex

ELIGIBLE = ("keep", "keep-simpler")
BENCH_ROWS = 200
BENCH_BATCH = 8
TEMPLATE = Path(__file__).parent / "templates" / "predict.py.tmpl"


# ---- request ----


def _is_pending(exp_dir: Path, run_id: str) -> bool:
    try:
        return store.read_request(exp_dir, run_id).get("status") not in store.TERMINAL
    except FileNotFoundError:
        return False


def _request(exp_dir: Path, run_id: str, seeds: int) -> str:
    if seeds < 1:
        raise ValueError("--seeds must be at least 1")
    with ex.state_lock(exp_dir):
        ex.verify_checksums(exp_dir, store.project_root(exp_dir))
        _, _, state = ex.load(exp_dir)
        if state.promoted or (exp_dir / "final" / "test_metrics.json").is_file():
            raise ValueError("already promoted; test is scored once")
        if state.promote_pending and _is_pending(exp_dir, state.promote_pending):
            raise ValueError(
                f"a promote is already pending ({state.promote_pending}); "
                f"wait for it with `finetune-kit wait {state.promote_pending} --exp {exp_dir.name}`"
            )
        src = store.read_request(exp_dir, run_id)
        verdict = state.finalized.get(run_id)
        if run_id != state.noise_run and verdict not in ELIGIBLE:
            raise ValueError(
                f"run {run_id} has verdict {verdict or src.get('status')!r}; only a keep or keep-simpler "
                f"run, or the noise run ({state.noise_run}), can be promoted"
            )
        rid = store.new_run(
            exp_dir, "promote", src.get("overrides") or {}, list(range(seeds)), f"promote {run_id}",
            extra={"source_run": run_id},
        )
        state = ex.load(exp_dir)[2]
        state.promote_pending = rid
        ex.save_state(exp_dir, state)
    return rid


def result(exp_dir: Path, run_id: str) -> dict:
    """Status of a promote run plus, once finalised, its promote payload."""
    req = store.read_request(exp_dir, run_id)
    return {"run_id": run_id, "status": req["status"], "source_run": req.get("source_run"), **req.get("promote", {})}


def promote(exp_dir: Path, run_id: str, seeds: int = 3, sync: bool = False) -> dict:
    """Queue the final retrain of `run_id` (returns at once), or run it in-process (`sync`, tests)."""
    from finetune_kit import queue

    exp_dir = Path(exp_dir).resolve()
    ex.require_setup(ex.load(exp_dir)[0])
    rid = _request(exp_dir, run_id, seeds)
    if not sync:
        position = queue.enqueue(exp_dir, rid)
        return {"run_id": rid, "status": "queued", "source_run": run_id, "position": position}
    try:
        queue._process(exp_dir, rid)
    except Exception as e:  # same last resort as the worker loop
        queue._mark_failed(exp_dir, rid, f"{type(e).__name__}: {e}")
    return result(exp_dir, rid)


# ---- model save / load ----


def _local_ref(ref):
    """`org/repo--module.Class` -> `module.Class` (also inside tokenizer [slow, fast] pairs)."""
    if isinstance(ref, (list, tuple)):
        return [_local_ref(r) for r in ref]
    return ref.split("--", 1)[1] if isinstance(ref, str) and "--" in ref else ref


def _localise_auto_map(path: Path, pins: dict[str, str]) -> None:
    if not path.is_file():
        return
    d = json.loads(path.read_text())
    if "auto_map" not in d and not pins:
        return
    d["auto_map"] = {**{k: _local_ref(v) for k, v in (d.get("auto_map") or {}).items()}, **pins}
    path.write_text(json.dumps(d, indent=2))


def _save_remote_code(base, model_dir: Path) -> None:
    """Copy the custom modeling/config code next to the weights so auto_map resolves offline."""
    from transformers.dynamic_module_utils import custom_object_save

    pins = {}
    for obj, auto in ((base, "AutoModel"), (base.config, "AutoConfig")):
        if not type(obj).__module__.startswith("transformers."):  # built-in classes need no code
            custom_object_save(obj, model_dir)
            pins[auto] = f"{type(obj).__module__.rsplit('.', 1)[-1]}.{type(obj).__name__}"
    _localise_auto_map(model_dir / "config.json", pins)
    _localise_auto_map(model_dir / "tokenizer_config.json", {})


def save_model(model, profile, model_dir: Path) -> None:
    """Backbone config + tokenizer (save_pretrained), remote code if any, and the classifier state dict."""
    import torch
    from transformers import AutoTokenizer

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    bb = model.backbone
    base = bb.get_base_model() if hasattr(bb, "get_base_model") else bb  # peft-wrapped with LoRA
    base.config.save_pretrained(model_dir)
    tok = AutoTokenizer.from_pretrained(
        profile.hf_id, revision=profile.revision, trust_remote_code=profile.trust_remote_code
    )
    tok.save_pretrained(model_dir)
    if profile.trust_remote_code:
        _save_remote_code(base, model_dir)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, model_dir / "classifier.pt")


def load_model(model_dir: Path, cfg, profile, n_labels: int, device: str):
    """Rebuild the classifier from a saved model dir (no hub download); returns (model, tokenizer)."""
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from finetune_kit.train.models import SeqClassifier, TokenClassifier, apply_lora

    kw = {"trust_remote_code": profile.trust_remote_code}
    backbone = AutoModel.from_config(AutoConfig.from_pretrained(model_dir, **kw), **kw)
    if cfg.lora_r > 0:
        backbone = apply_lora(backbone, profile, cfg.lora_r, cfg.lora_alpha)
    cls = TokenClassifier if cfg.task == "token-cls" else SeqClassifier
    model = cls(backbone, n_labels, cfg)
    model.load_state_dict(torch.load(Path(model_dir) / "classifier.pt", map_location="cpu", weights_only=True))
    tok = AutoTokenizer.from_pretrained(model_dir, **kw)
    return model.to(device).eval(), tok


def _encode_eval(tok, cfg, rows: list[dict], labels: list[str]) -> SimpleNamespace:
    """The val-side fields of `loop._Data`, so `loop._predict` can score any rows."""
    from finetune_kit.train.models import encode_seq, encode_tokens

    ns = SimpleNamespace(labels=labels, token=cfg.task == "token-cls", pad=tok.pad_token_id)
    if not ns.token:
        ns.val_ids = encode_seq(tok, [r["text"] for r in rows], cfg.max_len)[0]
        return ns
    tag2id = {t: i for i, t in enumerate(labels)}
    tags = [[t if t in tag2id else "O" for t in r["tags"]] for r in rows]
    ns.val_ids, ns.val_y, ns.val_words, _ = encode_tokens(
        tok, [r["tokens"] for r in rows], tags, tag2id, cfg.max_len, cfg.label_align,
    )
    ns.val_n_words = [len(r["tokens"]) for r in rows]
    return ns


# ---- throughput ----


def _sync(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def benchmark(model, tok, rows: list[dict], device: str, max_len: int = 512) -> dict:
    """End-to-end (tokenise + forward) speed on up to 200 rows at batch 8, after 1 warm-up batch."""
    import torch

    rows = rows[:BENCH_ROWS]
    if not rows:
        raise ValueError("no rows to benchmark")
    model.to(device).eval()
    token = "tokens" in rows[0]
    batches = [rows[i : i + BENCH_BATCH] for i in range(0, len(rows), BENCH_BATCH)]

    @torch.no_grad()
    def run(batch):
        inputs = [r["tokens"] for r in batch] if token else [r["text"] for r in batch]
        enc = tok(
            inputs, is_split_into_words=token, padding=True, truncation=True, max_length=max_len,
            return_tensors="pt",
        )
        model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        _sync(device)

    run(batches[0])  # warm-up
    times = []
    for b in batches:
        t0 = time.perf_counter()
        run(b)
        times.append(time.perf_counter() - t0)
    return {
        "rows_per_s": round(len(rows) / sum(times), 2),
        "p50_latency_ms": round(statistics.median(times) * 1000, 3),
        "n_rows": len(rows), "batch": BENCH_BATCH,
    }


def _dir_mb(path: Path) -> float:
    return round(sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) / 1e6, 3)


# ---- finish (worker side) ----


def at_threshold(truth, probs, pos: int, threshold: float) -> dict:
    """Binary decision `score >= threshold` for the positive class."""
    import numpy as np

    y = np.asarray(truth) == pos
    flag = np.asarray(probs, dtype=float)[:, pos] >= threshold
    tp = int((y & flag).sum())
    return {
        "threshold": threshold,
        "recall": tp / int(y.sum()) if y.any() else None,
        "precision": tp / int(flag.sum()) if flag.any() else None,
        "skip": float((~flag).mean()),
        "accuracy": float((flag == y).mean()),
    }


def _threshold(exp, exp_dir: Path, labels: list[str], pos: int, truth, probs) -> dict | None:
    """Threshold on val for the first recall/precision-constrained metric (objective, then guardrail)."""
    from finetune_kit import metrics
    from finetune_kit.train import worker

    for ref in (exp.objective, exp.guardrail):
        if ref is None:
            continue
        m = worker._metric(exp, ref, exp_dir)
        if not m.constrained:
            continue
        t = metrics.threshold_for(m, truth, probs, ref.params, pos)
        if t is None:
            return None
        return {
            "threshold": t, "metric": ref.metric, "params": {**m.params, **ref.params},
            "constraint": m.constrained, "positive_class": labels[pos], "positive_index": pos,
        }
    return None


def _context(exp_dir: Path, req: dict) -> SimpleNamespace:
    import numpy as np

    from finetune_kit.train import worker
    from finetune_kit.train.config import build_config

    exp, _, _ = ex.load(exp_dir)
    task = exp.task
    overrides = dict(req.get("overrides") or {})
    profile = store.profile_for(exp, overrides, store.project_root(exp_dir))
    overrides.pop("model", None)
    labels = data.label_map(data.load_rows(exp_dir / "splits" / "train.jsonl", task), task)
    pos = worker.positive_index(exp, labels)
    direction, _ = store.metric_info(exp, exp.objective)
    ms = [m for m in store.seed_metrics(exp_dir, req["run_id"]) if m.get("objective") is not None]
    if not ms:
        raise ValueError(f"promote run {req['run_id']} has no finished seeds")
    best = (max if direction == "max" else min)(ms, key=lambda m: m["objective"])
    sdir = store.run_dir(exp_dir, req["run_id"]) / f"seed_{best['seed']}"
    val_rows = data.load_rows(exp_dir / "splits" / "val.jsonl", task)
    val_truth = worker.truth_for(task, labels, val_rows, "val")
    if task == "seq-cls":
        val_out = np.load(sdir / "val_preds.npy")
        threshold = _threshold(exp, exp_dir, labels, pos, val_truth, val_out)
    else:
        val_out, threshold = json.loads((sdir / "val_preds.json").read_text()), None
    final = exp_dir / "final"
    return SimpleNamespace(
        exp=exp, task=task, profile=profile, labels=labels, pos=pos, seed=best["seed"], seed_metrics=best,
        cfg=build_config(overrides, profile, task, best["seed"]), val_rows=val_rows, val_truth=val_truth,
        val_out=val_out, threshold=threshold, final=final,
        model_src=sdir / "model" if (sdir / "model").is_dir() else final / "model",
        model=None, tok=None,
    )


def _ensure_model(ctx, device: str) -> None:
    if ctx.model is None:
        ctx.model, ctx.tok = load_model(ctx.model_src, ctx.cfg, ctx.profile, len(ctx.labels), device)


# constrained objective -> the honest at_threshold field it corresponds to
_AT_FIELD = {"skip_at_recall": "skip", "precision_at_recall": "precision", "recall_at_precision": "recall"}


def _label_test(exp_dir: Path, ctx, scored: dict) -> None:
    """A constrained objective on test re-fits its threshold on test labels (in-sample): rename it,
    and point `headline` at the val-threshold number instead."""
    from finetune_kit.train import worker

    name = ctx.exp.objective.metric
    if not worker._metric(ctx.exp, ctx.exp.objective, exp_dir).constrained:
        scored["headline"] = {"metric": name, "source": "objective", "value": scored["objective"]}
        return
    scored["objective_oracle_threshold"] = scored.pop("objective")
    field = _AT_FIELD.get(name)
    at = scored.get("at_threshold")
    if at is not None and field is not None:
        scored["headline"] = {"metric": name, "source": f"at_threshold.{field}", "value": at[field]}
    else:  # no val threshold reached the target: only the in-sample number exists
        scored["headline"] = {
            "metric": name, "source": "objective_oracle_threshold", "value": scored["objective_oracle_threshold"],
            "note": "no threshold on val met the constraint; this value uses a threshold fitted on test",
        }


def _mark_promoted(exp_dir: Path) -> None:
    with ex.state_lock(exp_dir):
        state = ex.load(exp_dir)[2]
        state.promoted, state.promote_pending = True, None
        ex.save_state(exp_dir, state)


def _score_test(exp_dir: Path, req: dict, ctx) -> dict:
    """The single read and scoring of the test split.

    test_metrics.json is written before `state.promoted` is set: a crash in between leaves the
    file, which a resumed finish (and any new promote request) treats as the test being used.
    """
    from finetune_kit.train import worker
    from finetune_kit.train.loop import _predict, pick_device, runtime_config

    device = pick_device()
    _ensure_model(ctx, device)
    rows = data.read_test(ctx.exp.uid)
    cfg = runtime_config(ctx.cfg, ctx.profile, device)
    out = _predict(ctx.model, cfg, _encode_eval(ctx.tok, cfg, rows, ctx.labels), device)
    truth = worker.truth_for(ctx.task, ctx.labels, rows, "test")
    scored = {
        "promote_run": req["run_id"], "source_run": req.get("source_run"), "seed": ctx.seed,
        "n_rows": len(rows), "objective_metric": ctx.exp.objective.metric,
        "guardrail_metric": ctx.exp.guardrail.metric if ctx.exp.guardrail else None,
        **worker.score_all(ctx.exp, exp_dir, ctx.labels, ctx.pos, truth, out),
    }
    if ctx.threshold is not None:
        scored["at_threshold"] = at_threshold(truth, out, ctx.pos, ctx.threshold["threshold"])
    _label_test(exp_dir, ctx, scored)
    ctx.final.mkdir(exist_ok=True)
    store.write_json(ctx.final / "test_metrics.json", scored)
    _mark_promoted(exp_dir)  # the test is consumed now, whatever happens next
    return scored


def _previous_test_metrics(exp_dir: Path, req: dict, ctx) -> dict | None:
    """test_metrics.json of this run if the test was already scored (resume), else None."""
    p = ctx.final / "test_metrics.json"
    promoted = ex.load(exp_dir)[2].promoted
    if not p.is_file():
        if promoted:
            raise RuntimeError("experiment is promoted but final/test_metrics.json is missing; the test is not rescored")
        return None
    scored = json.loads(p.read_text())
    if not promoted:
        _mark_promoted(exp_dir)  # scored before a crash/failure that skipped the state update
    if scored.get("promote_run") != req["run_id"]:
        raise ValueError(f"already promoted by {scored.get('promote_run')}; test is scored once")
    return scored


def _render_predict(exp_dir: Path, req: dict) -> str:
    import finetune_kit

    return (
        TEMPLATE.read_text()
        .replace("{{EXPERIMENT}}", exp_dir.name)
        .replace("{{RUN}}", req["run_id"])
        .replace("{{VERSION}}", finetune_kit.__version__)
    )


def _export(exp_dir: Path, req: dict, ctx) -> dict:
    from finetune_kit.profiles import to_dict
    from finetune_kit.train import worker
    from finetune_kit.train.loop import pick_device

    final = ctx.final
    final.mkdir(exist_ok=True)
    dst = final / "model"
    if ctx.model_src != dst:
        tmp = final / ".model.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(ctx.model_src, tmp)
        shutil.rmtree(dst, ignore_errors=True)
        os.replace(tmp, dst)
    store.write_json(final / "recipe.json", {
        "task": ctx.task, "labels": ctx.labels, "positive_index": ctx.pos,
        "config": ctx.cfg.model_dump(mode="json"),
        "profile": {"key": ctx.profile.key, **to_dict(ctx.profile)},
        "versions": store.current_versions(ctx.profile),
        "objective": ctx.exp.objective.model_dump(),
        "guardrail": ctx.exp.guardrail.model_dump() if ctx.exp.guardrail else None,
        "experiment": exp_dir.name, "source_run": req.get("source_run"), "promote_run": req["run_id"],
        "seed": ctx.seed, "created": store.now(),
    })
    if ctx.threshold is not None:
        store.write_json(final / "threshold.json", ctx.threshold)
    else:
        (final / "threshold.json").unlink(missing_ok=True)

    keep = ("objective", "guardrail", "report", "report_errors", "train_s", "best_epoch", "device", "n_train")
    val_m = {"seed": ctx.seed, "n_rows": len(ctx.val_rows), **{k: ctx.seed_metrics.get(k) for k in keep}}
    if ctx.threshold is not None:
        val_m["at_threshold"] = at_threshold(ctx.val_truth, ctx.val_out, ctx.pos, ctx.threshold["threshold"])
    store.write_json(final / "val_metrics.json", val_m)

    trained_on = ctx.seed_metrics.get("device")
    device = trained_on if trained_on in ("cuda", "mps") else pick_device()
    _ensure_model(ctx, device)
    bench = {}
    for dev in dict.fromkeys([device, "cpu"]):
        bench[dev] = benchmark(ctx.model, ctx.tok, ctx.val_rows, dev, ctx.cfg.max_len)
    throughput = {
        "device": device,
        "rows_per_s": {d: b["rows_per_s"] for d, b in bench.items()},
        "p50_latency_ms": {d: b["p50_latency_ms"] for d, b in bench.items()},
        "n_rows": bench["cpu"]["n_rows"], "batch": BENCH_BATCH, "model_mb": _dir_mb(dst),
    }
    store.write_json(final / "throughput.json", throughput)
    (final / "predict.py").write_text(_render_predict(exp_dir, req))

    for sdir in store.run_dir(exp_dir, req["run_id"]).glob("seed_*"):  # no checkpoints outside final/
        shutil.rmtree(sdir / "model", ignore_errors=True)
    return {"val_metrics": val_m, "throughput": throughput}


def _clear_pending(exp_dir: Path, run_id: str) -> None:
    with ex.state_lock(exp_dir):
        state = ex.load(exp_dir)[2]
        if state.promote_pending == run_id:
            state.promote_pending = None
            ex.save_state(exp_dir, state)


def finish(exp_dir: Path, req: dict) -> dict:
    """Finalise a promote run (called by the worker; idempotent). Returns the promote payload.

    Any failure before the test is scored leaves `promoted` False (test unused). Once scored,
    `promoted` stays True even if exporting fails; a resumed finish never rescores the test
    (an existing final/test_metrics.json of this run counts as scored).
    """
    exp_dir = Path(exp_dir)
    out: dict = {"source_run": req.get("source_run")}
    try:
        if req["status"] != "done":
            raise RuntimeError(f"promote training ended with status {req['status']}: {req.get('error', 'see seed logs')}")
        ctx = _context(exp_dir, req)
        out.update(seed=ctx.seed, threshold=ctx.threshold)
        out["test_metrics"] = _previous_test_metrics(exp_dir, req, ctx) or _score_test(exp_dir, req, ctx)
        out["final"] = str(ctx.final)
    except Exception as e:
        _clear_pending(exp_dir, req["run_id"])
        return {**out, "error": f"{type(e).__name__}: {e}"}
    try:
        out.update(_export(exp_dir, req, ctx))
    except Exception as e:
        out["error"] = f"export failed after test scoring (test is consumed): {type(e).__name__}: {e}"
    finally:
        _clear_pending(exp_dir, req["run_id"])
    return out
