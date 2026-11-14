"""File-based run queue, the detached worker daemon, stale-run recovery and verdict bookkeeping.

Queue: `kit_home()/queue.jsonl`, one `{"exp_dir", "run_id"}` per line, FIFO. The head entry stays
in the file while it runs and is removed once the run is finalised, so a killed worker leaves
it in place for `recover_stale()`. All queue reads/writes and the worker-lock handshake happen
under an `fcntl.flock` on `kit_home()/queue.lock`.
"""
import fcntl
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from finetune_kit import experiment as ex
from finetune_kit import stats, store
from finetune_kit.catalog import validate_config
from finetune_kit.paths import kit_home, test_env

LONG_TIMEOUT_S = 4 * 3600
LONG_STAGES = ("bakeoff", "lr-scan", "noise")


def queue_path() -> Path:
    return kit_home() / "queue.jsonl"


def worker_lock_path() -> Path:
    return kit_home() / "worker.lock"


@contextmanager
def _qlock():
    kit_home().mkdir(parents=True, exist_ok=True)
    with open(kit_home() / "queue.lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _read() -> list[dict]:
    p = queue_path()
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _write(entries: list[dict]) -> None:
    kit_home().mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=kit_home(), prefix=".queue.")
    with os.fdopen(fd, "w") as f:
        f.write("".join(json.dumps(e) + "\n" for e in entries))
    os.replace(tmp, queue_path())


def read_queue() -> list[dict]:
    with _qlock():
        return _read()


def _remove(entry: dict) -> None:
    with _qlock():
        entries = _read()
        if entry in entries:
            entries.remove(entry)
            _write(entries)


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def worker_pid() -> int | None:
    """PID in worker.lock if that process is alive, else None."""
    try:
        pid = int(worker_lock_path().read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    return pid if pid_alive(pid) else None


def _spawn_daemon() -> int:
    log = open(kit_home() / "worker.log", "a")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "finetune_kit.cli", "_worker"],
            start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        )
    finally:
        log.close()
    return proc.pid


def _ensure_daemon_locked() -> int | None:
    """Caller holds the queue lock. Spawn a daemon if work is queued and no live worker exists."""
    if test_env("FINETUNE_KIT_NO_DAEMON") == "1" or not _read():
        return None
    pid = worker_pid()
    if pid is None:
        pid = _spawn_daemon()
        worker_lock_path().write_text(str(pid))
    return pid


def ensure_daemon() -> int | None:
    """Restart the worker if the queue has entries but its daemon died; returns the worker PID."""
    with _qlock():
        return _ensure_daemon_locked()


def enqueue(exp_dir: Path, run_id: str) -> int:
    """Append a run to the queue, start the daemon if none is alive; returns 1-based position."""
    entry = {"exp_dir": str(Path(exp_dir).resolve()), "run_id": run_id}
    with _qlock():
        entries = _read() + [entry]
        _write(entries)
        _ensure_daemon_locked()
        return len(entries)


def _kill_group(pid: int | None) -> None:
    """SIGKILL a per-seed training process group (each seed runs in its own session)."""
    if not pid:
        return
    try:
        os.killpg(int(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def recover_stale() -> list[dict]:
    """Queued runs left `running`/`finalizing` by a dead worker.

    `running`: the orphaned training group is killed, the run becomes `crash` and is retried once.
    `finalizing`: all seeds finished; bookkeeping is completed (idempotently), no retraining.
    """
    out = []
    with _qlock():
        entries = _read()
    for e in entries:
        exp_dir, run_id = Path(e["exp_dir"]), e["run_id"]
        try:
            req = store.read_request(exp_dir, run_id)
        except FileNotFoundError:
            continue
        status = req.get("status")
        if status not in ("running", "finalizing") or pid_alive(req.get("pid")):
            continue
        if status == "finalizing":
            req["status"] = req.pop("outcome", "crash")
            _finalize(exp_dir, req, req.get("seed_status") or [])
            _remove(e)
            out.append({"exp_dir": str(exp_dir), "run_id": run_id, "status": req["status"], "retried": False})
            continue
        if pid_alive(req.get("child_pid")):
            _kill_group(req["child_pid"])
        attempt = {"status": "crash", "reason": f"worker (pid {req.get('pid')}) died mid-run", "at": store.now()}
        req.setdefault("attempts", []).append(attempt)
        retried = req.get("retries", 0) == 0
        req.update(pid=None, child_pid=None)
        if retried:
            req.update(status="queued", retries=req.get("retries", 0) + 1)
            store.write_request(exp_dir, run_id, req)
        else:
            req.update(status="crash", error=attempt["reason"])
            _finalize(exp_dir, req, req.get("seed_status") or [])
            _remove(e)
        out.append({"exp_dir": str(exp_dir), "run_id": run_id, "status": "crash", "retried": retried})
    return out


def _acquire_worker_lock() -> bool:
    me = os.getpid()
    with _qlock():
        pid = worker_pid()
        if pid is not None and pid != me:
            return False
        worker_lock_path().write_text(str(me))
        return True


def _release_worker_lock() -> None:
    try:
        if int(worker_lock_path().read_text().strip()) == os.getpid():
            worker_lock_path().unlink()
    except (FileNotFoundError, ValueError):
        pass


def worker_loop(once: bool = False) -> None:
    """Run queued runs one at a time (FIFO) until the queue is empty."""
    if not _acquire_worker_lock():
        return
    try:
        recover_stale()
        while True:
            with _qlock():
                entries = _read()
                if not entries:
                    _release_worker_lock()  # atomically with the empty check, see enqueue()
                    return
                head = entries[0]
            try:
                _process(Path(head["exp_dir"]), head["run_id"])
            except Exception as e:  # never let one bad run wedge the queue
                print(f"worker: {head}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                _mark_failed(Path(head["exp_dir"]), head["run_id"], f"{type(e).__name__}: {e}")
            _remove(head)
            if once:
                return
    finally:
        _release_worker_lock()


def _mark_failed(exp_dir: Path, run_id: str, error: str) -> None:
    """Last resort after an unexpected worker error: still count the run, else at least mark it."""
    try:
        req = store.read_request(exp_dir, run_id)
    except Exception:
        return
    if req.get("status") in store.TERMINAL:
        return
    if req.get("status") == "finalizing":  # seeds finished; keep their outcome, don't force crash
        req["status"] = req.pop("outcome", "crash")
    else:
        req.update(status="crash", error=error)
    try:
        _finalize(exp_dir, req, req.get("seed_status") or [])
    except Exception:
        try:
            finalized = ex.load(exp_dir)[2].finalized
        except Exception:
            finalized = {}
        fallback = "discard" if req["stage"] == "research" else "crash"
        req["verdict"] = finalized.get(run_id, fallback)
        req.setdefault("error", error)
        req.pop("outcome", None)
        try:
            store.write_request(exp_dir, run_id, req)
        except Exception:
            pass


def timeout_for(exp_dir: Path, stage: str, state: ex.State) -> float:
    env = test_env("FINETUNE_KIT_TIMEOUT_S")
    if env:
        return float(env)
    if stage in LONG_STAGES or state.baseline_train_s is None:
        return LONG_TIMEOUT_S
    t = 3 * state.baseline_train_s
    if stage == "promote":
        cfg, _, _ = ex.load(exp_dir)
        n_full = _count_lines(Path(exp_dir) / "splits" / "train.jsonl")
        n_research = min(cfg.research_subset, n_full) if cfg.research_subset else n_full
        t *= max(1.0, n_full / max(n_research, 1))
    return t


def _objective(seed_dir: Path) -> float | None:
    try:
        return json.loads((seed_dir / "metrics.json").read_text()).get("objective")
    except (FileNotFoundError, ValueError):
        return None


def _count_lines(p: Path) -> int:
    if not p.is_file():
        return 0
    with p.open() as f:
        return sum(1 for line in f if line.strip())


def _precheck(exp_dir: Path, req: dict) -> str | None:
    """Re-check at execution time what `finetune-kit run` checked at enqueue time.

    request.json is agent-reachable, so what will execute (stage, overrides, seeds) must hash to
    the value recorded in state at enqueue; research overrides are also re-validated.
    """
    try:
        ex.verify_checksums(exp_dir, store.project_root(exp_dir))
    except ex.TamperError as e:
        return f"refused at execution: {e}"
    cfg, space, state = ex.load(exp_dir)
    if state.request_hashes.get(req["run_id"]) != store.request_hash(req):
        return "refused at execution: request.json was modified after enqueue"
    if req["stage"] != "research":
        return None
    if space is None:
        return "refused at execution: no search space committed"
    profile = store.profile_for(cfg, {}, store.project_root(exp_dir))
    overrides = dict(req.get("overrides") or {})
    agent = req.get("config") or {}
    if "lr" in overrides:
        lr = overrides["lr"]
        if lr != agent.get("lr", state.lr_scan_best):
            return f"refused at execution: lr {lr!r} is neither the configured value nor the LR-scan result"
        if "lr" not in agent:
            overrides.pop("lr")  # added by the CLI from the LR scan, not chosen by the agent
    errs = validate_config(
        overrides, space.tunable, space.losses, store.effective_defaults(cfg, state, profile), cfg.task, profile,
    )
    return "refused at execution: " + "; ".join(errs) if errs else None


def _run_seed(exp_dir: Path, req: dict, seed: int, timeout: float) -> str:
    sdir = store.run_dir(exp_dir, req["run_id"]) / f"seed_{seed}"
    sdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "finetune_kit.train.worker",
           "--exp", str(exp_dir), "--run", req["run_id"], "--seed", str(seed)]
    with open(sdir / "train.log", "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        req["child_pid"] = proc.pid
        store.write_request(exp_dir, req["run_id"], req)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc.pid)
            proc.wait()
            return "timeout"
        finally:
            req["child_pid"] = None
    return "done" if rc == 0 and _objective(sdir) is not None else "crash"


def _process(exp_dir: Path, run_id: str) -> None:
    try:
        req = store.read_request(exp_dir, run_id)
    except FileNotFoundError:
        return
    if req.get("status") != "queued":
        return
    rdir = store.run_dir(exp_dir, run_id)
    for old in rdir.glob("seed_*"):
        shutil.rmtree(old)
    req.update(status="running", pid=os.getpid(), started=store.now(), child_pid=None)
    store.write_request(exp_dir, run_id, req)

    seeds_out: list[dict] = []
    problem = _precheck(exp_dir, req)
    if problem:
        req["error"] = problem
    else:
        timeout = timeout_for(exp_dir, req["stage"], ex.load(exp_dir)[2])
        for seed in req["seeds"]:
            t0 = time.monotonic()
            status = _run_seed(exp_dir, req, seed, timeout)
            seeds_out.append({"seed": seed, "status": status, "wall_s": round(time.monotonic() - t0, 3)})
            if status != "done":
                break  # rule 4: a crash or timeout discards the run; don't burn time on more seeds
    req["seed_status"] = seeds_out
    req["status"] = "crash" if problem else next(
        (s["status"] for s in seeds_out if s["status"] != "done"), "done"
    )
    req["finished"] = store.now()
    _finalize(exp_dir, req, seeds_out)


def _verdict(exp_dir: Path, cfg, state: ex.State, req: dict, summary, direction: str, min_delta: float) -> str:
    stage, ok = req["stage"], req["status"] == "done"
    if stage == "noise":
        if not ok:
            return req["status"]
        ms = store.seed_metrics(exp_dir, req["run_id"])
        guards = [m["guardrail"] for m in ms if m.get("guardrail") is not None]
        state.sigma_obj = stats.sigma([m["objective"] for m in ms])
        state.sigma_guard = stats.sigma(guards) if guards else None
        state.noise_run = state.best_run = req["run_id"]
        walls = [s["wall_s"] for s in req.get("seed_status") or []]
        state.baseline_train_s = statistics.fmean(walls)
        return "baseline"
    if stage == "research":
        if not (ok and summary is not None and state.best_run):
            return "discard"
        best = store.summarize(exp_dir, state.best_run)
        guard_dir = store.metric_info(cfg, cfg.guardrail)[0] if cfg.guardrail else "max"
        return stats.verdict(
            summary, best, state.sigma_obj or 0.0, state.sigma_guard, direction, min_delta, guard_dir,
        )
    return stage if ok else req["status"]


def _finalize(exp_dir: Path, req: dict, seeds_out: list[dict]) -> None:
    """Idempotent bookkeeping for a finished run.

    The request is first marked `finalizing` (with the intended outcome) so a worker killed
    mid-way is resumed by `recover_stale()` without retraining. State changes are applied once
    (`state.finalized[run_id]` is saved atomically with them), and the results row is appended
    only if absent.
    """
    run_id, stage = req["run_id"], req["stage"]
    req["seed_status"] = seeds_out
    store.write_request(exp_dir, run_id, {**req, "status": "finalizing", "outcome": req["status"]})
    wall = sum(s["wall_s"] for s in seeds_out)
    with ex.state_lock(exp_dir):
        cfg, _, state = ex.load(exp_dir)
        direction, metric_delta = store.metric_info(cfg, cfg.objective)
        min_delta = cfg.min_delta if cfg.min_delta is not None else metric_delta
        try:
            summary = store.summarize(exp_dir, run_id)
        except ValueError:
            summary = None
        if run_id in state.finalized:
            verdict = state.finalized[run_id]
        else:
            try:
                verdict = _verdict(exp_dir, cfg, state, req, summary, direction, min_delta)
            except Exception as e:  # a failed verdict still counts the run
                verdict = "discard" if stage == "research" else "crash"
                req["error"] = f"verdict failed: {type(e).__name__}: {e}"
            if stage == "research":
                if verdict in ("keep", "keep-simpler"):
                    state.best_run = run_id
                    state.non_keep_streak = 0
                else:
                    state.non_keep_streak += 1
                state.runs_used += 1
                state.hours_used += wall / 3600
            state.finalized[run_id] = verdict
            ex.save_state(exp_dir, state)

        if stage == "lr-scan":
            _finish_lr_scan(exp_dir, req, direction)
        elif stage == "bakeoff":
            _finish_bakeoff(exp_dir, req, direction, min_delta)
        elif stage == "promote":
            from finetune_kit import promote  # lazy: pulls in torch

            req["promote"] = promote.finish(exp_dir, req)

        if run_id not in store.result_run_ids(exp_dir):
            desc = req.get("description", "")
            if stage == "research" and req["status"] != "done":
                desc = f"{desc} [{req['status']}]".strip()
            store.append_result(exp_dir, {
                "run_id": run_id, "round": req.get("round"), "stage": stage,
                "objective_mean": summary.objective_mean if summary else None,
                "objective_std": summary.objective_std if summary else None,
                "guardrail_mean": summary.guardrail_mean if summary else None,
                "n_seeds": summary.n_seeds if summary else 0,
                "minutes": round(wall / 60, 2), "verdict": verdict, "description": desc,
                "config": json.dumps(req.get("overrides") or {}, sort_keys=True),
            })
    req["verdict"] = verdict
    req.pop("outcome", None)
    store.write_request(exp_dir, run_id, req)


def _group(exp_dir: Path, req: dict, key: str) -> list[dict] | None:
    """All requests sharing `req[key]` (this one with its final status), or None if any is pending."""
    gid = req.get(key)
    group = [
        req if r["run_id"] == req["run_id"] else r
        for r in store.list_requests(exp_dir) if gid is not None and r.get(key) == gid
    ]
    if not group or any(r["status"] not in store.TERMINAL for r in group):
        return None
    return group


def _objectives(exp_dir: Path, r: dict) -> list[float]:
    if r["status"] != "done":
        return []
    return [m["objective"] for m in store.seed_metrics(exp_dir, r["run_id"]) if m.get("objective") is not None]


def _finish_lr_scan(exp_dir: Path, req: dict, direction: str) -> None:
    group = _group(exp_dir, req, "scan_id")
    if group is None:
        return
    scored = [(statistics.fmean(o), r["overrides"]["lr"]) for r in group if (o := _objectives(exp_dir, r))]
    scored = [(v, lr) for v, lr in scored if math.isfinite(v)]
    if not scored:
        return
    pick = max if direction == "max" else min
    _, _, state = ex.load(exp_dir)
    state.lr_scan_best = pick(scored, key=lambda t: t[0])[1]
    ex.save_state(exp_dir, state)


def _finish_bakeoff(exp_dir: Path, req: dict, direction: str, min_delta: float) -> None:
    group = _group(exp_dir, req, "bakeoff_id")
    if group is None:
        return
    results, runs, failed = {}, {}, []
    for r in group:
        model = r["overrides"]["model"]
        runs[model] = r["run_id"]
        objs = _objectives(exp_dir, r)
        if not objs:
            failed.append(model)
            continue
        train_s = [m.get("train_s") or 0.0 for m in store.seed_metrics(exp_dir, r["run_id"])]
        results[model] = (objs, statistics.fmean(train_s))
    _, _, state = ex.load(exp_dir)
    if results:
        state.bakeoff = stats.bakeoff_rank(results, direction, min_delta)
    else:
        state.bakeoff = {"ranking": [], "winner": None, "clear": False, "reason": "every bake-off run failed"}
    state.bakeoff.update(runs=runs, failed=failed)
    ex.save_state(exp_dir, state)
