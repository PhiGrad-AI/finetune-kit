import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import NoReturn, Optional

import typer

import finetune_kit

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Noise-aware fine-tuning research kit.")


@app.callback()
def _main() -> None:
    """Keeps `app` a command group even while it has a single command."""


def emit(obj: dict) -> None:
    typer.echo(json.dumps(obj))


def fail(msg: str) -> NoReturn:
    typer.echo(json.dumps({"error": msg}))
    raise typer.Exit(2)


@app.command()
def version() -> None:
    """Print the installed version as JSON."""
    emit({"version": finetune_kit.__version__})


@contextmanager
def _errors():
    """Map library errors to one JSON error object and exit code 2."""
    from finetune_kit.experiment import TamperError

    try:
        yield
    except (FileNotFoundError, ValueError, KeyError, TamperError) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
        fail(str(msg))


def _exp(name: str) -> Path:
    from finetune_kit.paths import find_experiment

    return find_experiment(name).resolve()


def _verify(exp_dir: Path) -> None:
    from finetune_kit import experiment as ex
    from finetune_kit.store import project_root

    ex.verify_checksums(exp_dir, project_root(exp_dir))


def _pending(exp_dir: Path, stage: str) -> int:
    from finetune_kit import store

    return sum(
        1 for r in store.list_requests(exp_dir)
        if r["stage"] == stage and r["status"] not in store.TERMINAL
    )


@app.command()
def run(
    exp: str = typer.Argument(..., help="Experiment: number, name or path."),
    config: str = typer.Option("{}", "--config", help="JSON object of hyperparameter overrides."),
    seeds: Optional[int] = typer.Option(None, "--seeds", help="Seeds (default: experiment.yaml seeds)."),
    stage: str = typer.Option("research", "--stage", help="research | noise"),
    desc: str = typer.Option("", "--desc", help="One-line description of the idea tested."),
) -> None:
    """Validate a config and queue a run; returns immediately with the run id."""
    from finetune_kit import experiment as ex
    from finetune_kit import queue, store
    from finetune_kit.catalog import validate_config

    with _errors():
        if stage not in ("research", "noise"):
            raise ValueError(f"--stage must be research or noise, not {stage!r}")
        exp_dir = _exp(exp)
        _verify(exp_dir)
        cfg, space, state = ex.load(exp_dir)
        root = store.project_root(exp_dir)
        try:
            overrides = json.loads(config)
        except json.JSONDecodeError as e:
            raise ValueError(f"--config is not valid JSON: {e}") from None
        if not isinstance(overrides, dict):
            raise ValueError("--config must be a JSON object")
        profile = store.profile_for(cfg, {}, root)
        warnings = store.version_warnings(cfg, profile)
        defaults = store.effective_defaults(cfg, state, profile)
        n = cfg.seeds if seeds is None else seeds

        if stage == "noise":
            if n < 2:
                raise ValueError("the noise floor needs at least 2 seeds")
            if overrides:
                warnings.append(
                    "noise stage uses the default config; ignored keys: " + ", ".join(sorted(overrides))
                )
            overrides = {} if state.lr_scan_best is None else {"lr": state.lr_scan_best}
            first = not any(r["stage"] == "noise" for r in store.list_requests(exp_dir))
            if first:
                ex.record_checksums(exp_dir, root)
            state = ex.load(exp_dir)[2]
            state.model_locked = True
            ex.save_state(exp_dir, state)
            desc = desc or "noise floor (default config)"
        else:
            if state.sigma_obj is None:
                raise ValueError("no noise floor yet: run `finetune-kit run EXP --stage noise` and wait for it first")
            if space is None:
                raise ValueError("no search space committed: run `finetune-kit space commit` first")
            if n < cfg.seeds:
                raise ValueError(f"research runs use at least {cfg.seeds} seeds (experiment.yaml)")
            pending = _pending(exp_dir, "research")
            if state.runs_used + pending >= cfg.budget.max_runs:
                raise ValueError(
                    f"budget spent: {state.runs_used} runs used + {pending} queued of max_runs {cfg.budget.max_runs}"
                )
            if state.hours_used >= cfg.budget.max_hours:
                raise ValueError(
                    f"budget spent: {state.hours_used:.2f} h used of max_hours {cfg.budget.max_hours}"
                )
            errs = validate_config(overrides, space.tunable, space.losses, defaults, cfg.task, profile)
            if errs:
                raise ValueError("; ".join(errs))
            overrides = {
                **({"lr": state.lr_scan_best} if state.lr_scan_best is not None else {}),
                **overrides,
            }
        run_id = store.new_run(exp_dir, stage, overrides, list(range(n)), desc)
        position = queue.enqueue(exp_dir, run_id)
    emit({"run_id": run_id, "status": "queued", "position": position, "warnings": warnings})


@app.command("lr-scan")
def lr_scan(exp: str = typer.Argument(..., help="Experiment: number, name or path.")) -> None:
    """Queue 5 short runs (20% of the schedule, seed 0) at LRs log-spaced over the profile range."""
    from finetune_kit import experiment as ex
    from finetune_kit import queue, store

    with _errors():
        exp_dir = _exp(exp)
        _verify(exp_dir)
        cfg, _, _ = ex.load(exp_dir)
        profile = store.profile_for(cfg, {}, store.project_root(exp_dir))
        lo, hi = profile.lr_range
        lrs = [float(f"{lo * (hi / lo) ** (i / 4):.3g}") for i in range(5)]
        scan_id = store.now()
        ids = [
            store.new_run(
                exp_dir, "lr-scan", {"lr": lr, "step_fraction": 0.2}, [0], f"lr scan {lr:g}",
                extra={"scan_id": scan_id},
            )
            for lr in lrs
        ]
        for rid in ids:
            queue.enqueue(exp_dir, rid)
    emit({"run_ids": ids, "lrs": lrs, "status": "queued"})


@app.command()
def bakeoff(
    args: list[str] = typer.Argument(..., help="EXP, or: choose EXP MODEL"),
    models: Optional[str] = typer.Option(None, "--models", help="Comma-separated model profile keys."),
) -> None:
    """Queue a model bake-off (`bakeoff EXP --models a,b`), show its result (`bakeoff EXP`),
    or fix the experiment's model (`bakeoff choose EXP MODEL`)."""
    from finetune_kit import experiment as ex
    from finetune_kit import queue, store
    from finetune_kit.profiles import load_profiles

    with _errors():
        if args[0] == "choose":
            if len(args) != 3:
                raise ValueError("usage: finetune-kit bakeoff choose EXP MODEL")
            exp_dir = _exp(args[1])
            cfg, _, state = ex.load(exp_dir)
            if state.model_locked:
                raise ValueError(f"model is locked ({cfg.model}); the noise floor or a choice already fixed it")
            profile = store.profile_for(cfg, {"model": args[2]}, store.project_root(exp_dir))
            if cfg.task not in profile.tasks:
                raise ValueError(f"model {profile.key!r} does not support {cfg.task}")
            cfg.model = profile.key
            cfg.versions["model_revision"] = profile.revision
            ex._dump_yaml(exp_dir / "experiment.yaml", cfg)
            state.model_locked = True
            ex.save_state(exp_dir, state)
            emit({"model": profile.key, "revision": profile.revision, "model_locked": True})
            return
        if len(args) != 1:
            raise ValueError("usage: finetune-kit bakeoff EXP [--models a,b] | bakeoff choose EXP MODEL")
        exp_dir = _exp(args[0])
        cfg, _, state = ex.load(exp_dir)
        if models is None:
            emit({"bakeoff": state.bakeoff, "pending": _pending(exp_dir, "bakeoff")})
            return
        if state.model_locked:
            raise ValueError(f"model is locked ({cfg.model}); a bake-off is only possible before it is fixed")
        _verify(exp_dir)
        keys = [k.strip() for k in models.split(",") if k.strip()]
        if len(keys) < 2 or len(set(keys)) != len(keys):
            raise ValueError("--models needs at least two distinct model keys")
        profiles = load_profiles(store.project_root(exp_dir))
        for k in keys:
            if k not in profiles:
                raise ValueError(f"unknown model {k!r}; see `finetune-kit models`")
            if cfg.task not in profiles[k].tasks:
                raise ValueError(f"model {k!r} does not support {cfg.task}")
        bid = store.now()
        ids = [
            store.new_run(
                exp_dir, "bakeoff", {"model": k}, list(range(cfg.seeds)), f"bake-off {k}",
                extra={"bakeoff_id": bid},
            )
            for k in keys
        ]
        for rid in ids:
            queue.enqueue(exp_dir, rid)
    emit({"run_ids": ids, "models": keys, "status": "queued"})


def _wait_payload(exp_dir: Path, req: dict) -> dict:
    from finetune_kit import store

    rdir = store.run_dir(exp_dir, req["run_id"])
    seeds = []
    by_seed = {m["seed"]: m for m in store.seed_metrics(exp_dir, req["run_id"])}
    for st in req.get("seed_status") or [{"seed": s} for s in req["seeds"]]:
        s = st["seed"]
        item = {**st, **by_seed.get(s, {})}
        err = rdir / f"seed_{s}" / "error.txt"
        if err.is_file():
            item["error"] = err.read_text().splitlines()[0] if err.read_text() else ""
        seeds.append(item)
    out = {"run_id": req["run_id"], "status": req["status"], "stage": req["stage"], "seeds": seeds}
    if "verdict" in req:
        out["verdict"] = req["verdict"]
    return out


@app.command()
def wait(
    run_id: str = typer.Argument(..., help="Run id as printed by `finetune-kit run`, e.g. r007."),
    exp: str = typer.Option(..., "--exp", help="Experiment the run belongs to (number, name or path)."),
    timeout: float = typer.Option(540, "--timeout", help="Seconds to block before returning status running."),
) -> None:
    """Block until a run finishes (or --timeout); prints per-seed metrics and the verdict.

    Example: finetune-kit wait r007 --exp 001
    """
    from finetune_kit import store

    with _errors():
        exp_dir = _exp(exp)
        t0 = time.monotonic()
        while True:
            req = store.read_request(exp_dir, run_id)
            if req["status"] in store.TERMINAL:
                emit(_wait_payload(exp_dir, req))
                return
            elapsed = time.monotonic() - t0
            if elapsed >= timeout:
                emit({"run_id": run_id, "status": req["status"], "elapsed_s": round(elapsed, 1)})
                return
            time.sleep(min(2.0, timeout - elapsed))


@app.command()
def status(exp: Optional[str] = typer.Argument(None, help="Only show runs of this experiment.")) -> None:
    """Show the run queue, the running job and the worker PID."""
    from finetune_kit import queue, store

    with _errors():
        only = _exp(exp) if exp else None
        items, running = [], None
        for e in queue.read_queue():
            exp_dir = Path(e["exp_dir"])
            if only is not None and exp_dir != only:
                continue
            try:
                req = store.read_request(exp_dir, e["run_id"])
            except FileNotFoundError:
                continue
            item = {"experiment": exp_dir.name, "run_id": e["run_id"], "stage": req["stage"], "status": req["status"]}
            if req["status"] == "running":
                running = {**item, "started": req.get("started"), "pid": req.get("pid")}
            else:
                items.append(item)
    emit({"queue": items, "running": running, "worker_pid": queue.worker_pid()})


@app.command("_worker", hidden=True)
def _worker() -> None:
    """Detached queue worker (started by `run`; exits when the queue is empty)."""
    from finetune_kit import queue

    queue.worker_loop()


if __name__ == "__main__":
    app()
