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


def _requests(exp_dir: Path, stage: str, *, pending_only: bool = False, skip_failed: bool = False) -> list[dict]:
    from finetune_kit import store

    out = []
    for r in store.list_requests(exp_dir):
        if r["stage"] != stage:
            continue
        if pending_only and r["status"] in store.TERMINAL:
            continue
        if skip_failed and r["status"] in ("crash", "timeout"):
            continue
        out.append(r)
    return out


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
        try:
            agent_config = json.loads(config)
        except json.JSONDecodeError as e:
            raise ValueError(f"--config is not valid JSON: {e}") from None
        if not isinstance(agent_config, dict):
            raise ValueError("--config must be a JSON object")
        # Checks, budget and new_run under one lock: parallel `run` calls cannot overshoot.
        with ex.state_lock(exp_dir):
            _verify(exp_dir)
            cfg, space, state = ex.load(exp_dir)
            root = store.project_root(exp_dir)
            profile = store.profile_for(cfg, {}, root)
            warnings = store.version_warnings(cfg, profile)
            defaults = store.effective_defaults(cfg, state, profile)
            n = cfg.seeds if seeds is None else seeds
            extra: dict = {}

            if stage == "noise":
                if n < 2:
                    raise ValueError("the noise floor needs at least 2 seeds")
                prior = _requests(exp_dir, "noise", skip_failed=True)
                if prior:
                    raise ValueError(
                        f"the noise floor is measured once per experiment ({prior[0]['run_id']} is "
                        f"{prior[0]['status']}); it cannot be re-measured — start a new experiment to re-baseline"
                    )
                if _requests(exp_dir, "lr-scan", pending_only=True):
                    raise ValueError("an LR scan is still pending; wait for it before measuring the noise floor")
                if agent_config:
                    warnings.append(
                        "noise stage uses the default config; ignored keys: " + ", ".join(sorted(agent_config))
                    )
                overrides = {} if state.lr_scan_best is None else {"lr": state.lr_scan_best}
                if not _requests(exp_dir, "noise"):
                    ex.record_checksums(exp_dir, root)
                state = ex.load(exp_dir)[2]
                state.model_locked = True
                ex.save_state(exp_dir, state)
                desc = desc or "noise floor (default config)"
            else:
                if state.sigma_obj is None:
                    raise ValueError(
                        "no noise floor yet: run `finetune-kit run EXP --stage noise` and wait for it first"
                    )
                if space is None:
                    raise ValueError("no search space committed: run `finetune-kit space commit` first")
                if n < cfg.seeds:
                    raise ValueError(f"research runs use at least {cfg.seeds} seeds (experiment.yaml)")
                pending = len(_requests(exp_dir, "research", pending_only=True))
                if state.runs_used + pending >= cfg.budget.max_runs:
                    raise ValueError(
                        f"budget spent: {state.runs_used} runs used + {pending} queued of max_runs "
                        f"{cfg.budget.max_runs}"
                    )
                if state.hours_used >= cfg.budget.max_hours:
                    raise ValueError(
                        f"budget spent: {state.hours_used:.2f} h used of max_hours {cfg.budget.max_hours}"
                    )
                errs = validate_config(agent_config, space.tunable, space.losses, defaults, cfg.task, profile)
                if errs:
                    raise ValueError("; ".join(errs))
                overrides = {
                    **({"lr": state.lr_scan_best} if state.lr_scan_best is not None else {}),
                    **agent_config,
                }
                extra = {"config": agent_config}  # re-validated by the worker at execution time
            run_id = store.new_run(exp_dir, stage, overrides, list(range(n)), desc, extra=extra)
        position = queue.enqueue(exp_dir, run_id)
    emit({"run_id": run_id, "status": "queued", "position": position, "warnings": warnings})


@app.command("lr-scan")
def lr_scan(exp: str = typer.Argument(..., help="Experiment: number, name or path.")) -> None:
    """Queue 5 short runs (20% of the schedule, seed 0) at LRs log-spaced over the profile range.

    Only before the noise floor: afterwards the default LR is fixed for the experiment.
    """
    from finetune_kit import experiment as ex
    from finetune_kit import queue, store

    with _errors():
        exp_dir = _exp(exp)
        with ex.state_lock(exp_dir):
            _verify(exp_dir)
            cfg, _, _ = ex.load(exp_dir)
            noise = _requests(exp_dir, "noise", skip_failed=True)
            if noise:
                raise ValueError(
                    f"the noise floor ({noise[0]['run_id']}) already fixed the default LR; an LR scan is "
                    "only possible before it — start a new experiment to rescan"
                )
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
            with ex.state_lock(exp_dir):
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
            emit({"bakeoff": state.bakeoff, "pending": len(_requests(exp_dir, "bakeoff", pending_only=True))})
            return
        with ex.state_lock(exp_dir):
            cfg, _, state = ex.load(exp_dir)
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
    if "promote" in req:
        out["promote"] = req["promote"]
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
    from finetune_kit import queue, store

    with _errors():
        exp_dir = _exp(exp)
        t0 = time.monotonic()
        while True:
            queue.ensure_daemon()  # restarts a dead worker if runs are still queued
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
        pid = queue.ensure_daemon() or queue.worker_pid()  # restarts a dead worker
    emit({"queue": items, "running": running, "worker_pid": pid})


@app.command()
def new(
    name: str = typer.Argument(..., help="Experiment name."),
    task: str = typer.Option(..., "--task", help="seq-cls | token-cls"),
    model: str = typer.Option(..., "--model", help="Model profile key (see `models`)."),
    train: str = typer.Option(..., "--train", help="Train JSONL."),
    val: Optional[str] = typer.Option(None, "--val", help="Val JSONL (with --test)."),
    test: Optional[str] = typer.Option(None, "--test", help="Test JSONL (goes to the vault)."),
    metric: str = typer.Option(..., "--metric", help="Objective metric."),
    metric_params: str = typer.Option("{}", "--metric-params", help="JSON object."),
    guardrail: Optional[str] = typer.Option(None, "--guardrail", help="Guardrail metric."),
    guardrail_params: str = typer.Option("{}", "--guardrail-params", help="JSON object."),
    positive_class: Optional[str] = typer.Option(None, "--positive-class"),
    custom_metric: Optional[str] = typer.Option(None, "--custom-metric", help="PATH:FUNC (seq-cls)."),
    direction: Optional[str] = typer.Option(None, "--direction", help="max | min (custom metric)."),
    seeds: int = typer.Option(3, "--seeds"),
    max_runs: int = typer.Option(40, "--max-runs"),
    max_hours: float = typer.Option(24, "--max-hours"),
    subset: Optional[int] = typer.Option(None, "--subset", help="Research train subset size."),
) -> None:
    """Create experiments/NNN-name with its config and recorded versions."""
    from finetune_kit import commands as cmd

    with _errors():
        d = cmd.setup_experiment(
            Path.cwd(), name, task=task, model=model, train=train, val=val, test=test, metric=metric,
            metric_params=cmd.parse_params(metric_params, "--metric-params"), guardrail=guardrail,
            guardrail_params=cmd.parse_params(guardrail_params, "--guardrail-params"),
            positive_class=positive_class, seeds=seeds, max_runs=max_runs, max_hours=max_hours,
            subset=subset, custom_metric=custom_metric, direction=direction,
        )
    emit({"experiment": d.name, "path": str(d)})


models_app = typer.Typer(add_completion=False, invoke_without_command=True, help="List or add model profiles.")
app.add_typer(models_app, name="models")


@models_app.callback()
def models(ctx: typer.Context, task: Optional[str] = typer.Option(None, "--task", help="Filter by task.")) -> None:
    """List model profiles (`models add HF_ID` registers a new one)."""
    from finetune_kit.profiles import load_profiles

    if ctx.invoked_subcommand is not None:
        return
    with _errors():
        profs = load_profiles(Path.cwd())
    emit({"models": [
        {"key": p.key, "hf_id": p.hf_id, "tasks": p.tasks, "max_len": p.max_len, "lr_default": p.lr_default,
         "lr_range": list(p.lr_range), "revision": p.revision, "notes": p.notes}
        for p in profs.values() if task is None or task in p.tasks
    ]})


@models_app.command("add")
def models_add(
    hf_id: str = typer.Argument(..., help="Hugging Face model id."),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code"),
) -> None:
    """Register a generic profile in .finetune-kit/models.yaml (config.json only, no weights)."""
    from finetune_kit.profiles import to_dict, add_profile

    with _errors():
        p = add_profile(Path.cwd(), hf_id, trust_remote_code)
    emit({"key": p.key, **to_dict(p)})


@app.command("metrics")
def metrics_cmd(task: Optional[str] = typer.Option(None, "--task", help="Filter by task.")) -> None:
    """List the built-in metrics."""
    from finetune_kit import metrics

    ms = [m for t in (["seq-cls", "token-cls"] if task is None else [task]) for m in metrics.list_for(t)]
    seen = {}
    for m in ms:
        seen[m.name] = m
    emit({"metrics": [
        {"name": m.name, "tasks": list(m.tasks), "direction": m.direction, "min_delta": m.min_delta,
         "params": m.params, "needs_positive": m.needs_positive, "constrained": m.constrained}
        for m in seen.values()
    ]})


@app.command("check-data")
def check_data(
    exp: str = typer.Argument(..., help="Experiment: number, name or path."),
    resplit: bool = typer.Option(False, "--resplit", help="Discard frozen splits and reset state."),
    val_frac: float = typer.Option(0.1, "--val-frac", help="Validation share when splitting (test stays 0.1)."),
) -> None:
    """Validate data, split and freeze it (test goes to the vault), write data-report.json."""
    from finetune_kit import commands as cmd

    with _errors():
        report = cmd.check_data(_exp(exp), resplit=resplit, val_frac=val_frac)
    emit(report)


@app.command()
def catalog(exp: str = typer.Argument(..., help="Experiment: number, name or path.")) -> None:
    """List tunable hyperparameters with recommendations for this experiment's data."""
    from finetune_kit import experiment as ex
    from finetune_kit import store
    from finetune_kit.catalog import listing

    with _errors():
        exp_dir = _exp(exp)
        cfg, _, _ = ex.load(exp_dir)
        rp = exp_dir / "data-report.json"
        report = json.loads(rp.read_text()) if rp.is_file() else {}
        profile = store.profile_for(cfg, {}, store.project_root(exp_dir))
        emit({"catalog": listing(cfg.task, profile, report, cfg.objective.metric)})


space_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Search space.")
app.add_typer(space_app, name="space")


@space_app.command("commit")
def space_commit(
    exp: str = typer.Argument(..., help="Experiment: number, name or path."),
    tunable: str = typer.Option(..., "--tunable", help="Comma-separated hyperparameter names."),
    losses: str = typer.Option("ce", "--losses", help="Comma-separated allowed losses."),
) -> None:
    """Commit (or re-commit, next round) the tunable hyperparameters and losses."""
    from finetune_kit import experiment as ex

    with _errors():
        exp_dir = _exp(exp)
        space = ex.SearchSpace(
            round=0, tunable=[t.strip() for t in tunable.split(",") if t.strip()],
            losses=[l.strip() for l in losses.split(",") if l.strip()],
        )
        with ex.state_lock(exp_dir):
            state = ex.commit_space(exp_dir, space)
    emit({"round": state.round, "tunable": space.tunable, "losses": space.losses})


@app.command()
def compare(exp: str = typer.Argument(..., help="Experiment: number, name or path.")) -> None:
    """Best run, noise floor, non-keep streak, budget left, stop flag and every result row."""
    from finetune_kit import commands as cmd

    with _errors():
        emit(cmd.compare(_exp(exp)))


@app.command()
def misses(
    exp: str = typer.Argument(..., help="Experiment: number, name or path."),
    run_id: str = typer.Argument(..., help="Run id, e.g. r007."),
    k: int = typer.Option(20, "--k", help="Rows per list."),
) -> None:
    """Worst val errors of the run's best seed (val only, never the test vault)."""
    from finetune_kit import commands as cmd

    with _errors():
        emit(cmd.misses(_exp(exp), run_id, k))


@app.command()
def report(exp: str = typer.Argument(..., help="Experiment: number, name or path.")) -> None:
    """Write experiments/NNN/report.html (self-contained) and print its path."""
    from finetune_kit import report as rep

    with _errors():
        path = rep.write_report(_exp(exp))
    emit({"path": str(path)})


@app.command()
def promote(
    exp: str = typer.Argument(..., help="Experiment: number, name or path."),
    run_id: str = typer.Argument(..., help="A keep / keep-simpler run, or the noise run."),
    seeds: int = typer.Option(3, "--seeds", help="Seeds for the final retrain on the full train split."),
) -> None:
    """Queue the final retrain; when it finishes the test split is scored once and final/ is written.

    Returns at once; `finetune-kit wait RUN --exp EXP` prints test metrics, threshold and throughput.
    """
    from finetune_kit import promote as pr

    with _errors():
        emit(pr.promote(_exp(exp), run_id, seeds))


@app.command()
def init(
    project: Optional[str] = typer.Argument(None, help="New project directory (default: the current one)."),
    ai: str = typer.Option(..., "--ai", help="claude | codex | cursor"),
    here: bool = typer.Option(False, "--here", help="Install into the current directory."),
    force: bool = typer.Option(False, "--force", help="Also replace an edited .finetune-kit/memory/research-rules.md."),
) -> None:
    """Install the finetune-kit stage commands, rules and templates for a coding agent."""
    from finetune_kit import installer

    with _errors():
        if project and here:
            raise ValueError("give a project directory or --here, not both")
        root = Path(project) if project else Path.cwd()
        root.mkdir(parents=True, exist_ok=True)
        written = installer.install(root, ai, force)
    agent = installer.AGENTS[ai]
    typer.echo(f"finetune-kit installed for {agent.label} in {root.resolve()} ({len(written)} files written).")
    if installer.rules_kept(root):
        typer.echo(f"Kept your edited {installer.RULES} (use --force to replace it with the shipped rules).")
    typer.echo(f"Next: open {agent.label} in this project and run {agent.invoke}setup")


@app.command("_guard", hidden=True)
def _guard() -> None:
    """Claude Code PreToolUse hook: exit 2 for edits to files only finetune-kit may write."""
    from finetune_kit import guard

    raise typer.Exit(guard.main())


@app.command("_worker", hidden=True)
def _worker() -> None:
    """Detached queue worker (started by `run`; exits when the queue is empty)."""
    from finetune_kit import queue

    queue.worker_loop()


if __name__ == "__main__":
    app()
