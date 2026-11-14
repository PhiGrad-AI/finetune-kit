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
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from finetune_kit import experiment as ex
from finetune_kit import stats, store
from finetune_kit.paths import kit_home

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


def enqueue(exp_dir: Path, run_id: str) -> int:
    """Append a run to the queue, start the daemon if none is alive; returns 1-based position."""
    entry = {"exp_dir": str(Path(exp_dir).resolve()), "run_id": run_id}
    with _qlock():
        entries = _read() + [entry]
        _write(entries)
        if os.environ.get("FINETUNE_KIT_NO_DAEMON") != "1" and worker_pid() is None:
            worker_lock_path().write_text(str(_spawn_daemon()))
        return len(entries)


def recover_stale() -> list[dict]:
    """Queued runs left `running` by a dead worker become `crash`; each is retried once."""
    out = []
    with _qlock():
        entries = _read()
    for e in entries:
        exp_dir, run_id = Path(e["exp_dir"]), e["run_id"]
        try:
            req = store.read_request(exp_dir, run_id)
        except FileNotFoundError:
            continue
        if req.get("status") != "running" or pid_alive(req.get("pid")):
            continue
        attempt = {"status": "crash", "reason": f"worker (pid {req.get('pid')}) died mid-run", "at": store.now()}
        req.setdefault("attempts", []).append(attempt)
        retried = req.get("retries", 0) == 0
        if retried:
            req.update(status="queued", retries=req.get("retries", 0) + 1, pid=None)
            store.write_request(exp_dir, run_id, req)
        else:
            req.update(status="crash", pid=None)
            _finalize(exp_dir, req, [])
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
    try:
        req = store.read_request(exp_dir, run_id)
        if req.get("status") not in store.TERMINAL:
            req.update(status="crash", verdict="discard" if req["stage"] == "research" else "crash", error=error)
            store.write_request(exp_dir, run_id, req)
    except Exception:
        pass


def timeout_for(exp_dir: Path, stage: str, state: ex.State) -> float:
    env = os.environ.get("FINETUNE_KIT_TIMEOUT_S")
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


def _process(exp_dir: Path, run_id: str) -> None:
    try:
        req = store.read_request(exp_dir, run_id)
    except FileNotFoundError:
        return
    if req.get("status") != "queued":
        return
    _, _, state = ex.load(exp_dir)
    rdir = store.run_dir(exp_dir, run_id)
    for old in rdir.glob("seed_*"):
        shutil.rmtree(old)
    req.update(status="running", pid=os.getpid(), started=store.now())
    store.write_request(exp_dir, run_id, req)

    timeout = timeout_for(exp_dir, req["stage"], state)
    seeds_out = []
    for seed in req["seeds"]:
        sdir = rdir / f"seed_{seed}"
        sdir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "finetune_kit.train.worker",
               "--exp", str(exp_dir), "--run", run_id, "--seed", str(seed)]
        t0 = time.monotonic()
        with open(sdir / "train.log", "w") as log:
            try:
                rc = subprocess.run(cmd, timeout=timeout, stdout=log, stderr=subprocess.STDOUT).returncode
                status = "done" if rc == 0 and _objective(sdir) is not None else "crash"
            except subprocess.TimeoutExpired:
                status = "timeout"
        seeds_out.append({"seed": seed, "status": status, "wall_s": round(time.monotonic() - t0, 3)})
        if status != "done":
            break  # rule 4: a crash or timeout discards the run; don't burn time on more seeds
    req["seed_status"] = seeds_out
    req["status"] = next((s["status"] for s in seeds_out if s["status"] != "done"), "done")
    req["finished"] = store.now()
    _finalize(exp_dir, req, seeds_out)


def _finalize(exp_dir: Path, req: dict, seeds_out: list[dict]) -> None:
    """Update state for the finished run, append its results row, then write the final request."""
    run_id, stage, ok = req["run_id"], req["stage"], req["status"] == "done"
    cfg, _, state = ex.load(exp_dir)
    direction, metric_delta = store.metric_info(cfg, cfg.objective)
    min_delta = cfg.min_delta if cfg.min_delta is not None else metric_delta
    wall = sum(s["wall_s"] for s in seeds_out)
    try:
        summary = store.summarize(exp_dir, run_id)
    except ValueError:
        summary = None

    if stage == "noise":
        verdict = "baseline" if ok else req["status"]
        if ok:
            ms = store.seed_metrics(exp_dir, run_id)
            guards = [m["guardrail"] for m in ms if m.get("guardrail") is not None]
            state.sigma_obj = stats.sigma([m["objective"] for m in ms])
            state.sigma_guard = stats.sigma(guards) if guards else None
            state.noise_run = state.best_run = run_id
            state.baseline_train_s = wall / len(seeds_out)
    elif stage == "research":
        verdict = "discard"
        if ok and summary is not None and state.best_run:
            best = store.summarize(exp_dir, state.best_run)
            guard_dir = store.metric_info(cfg, cfg.guardrail)[0] if cfg.guardrail else "max"
            verdict = stats.verdict(
                summary, best, state.sigma_obj or 0.0, state.sigma_guard, direction, min_delta, guard_dir,
            )
        if verdict in ("keep", "keep-simpler"):
            state.best_run = run_id
            state.non_keep_streak = 0
        else:
            state.non_keep_streak += 1
        state.runs_used += 1
        state.hours_used += wall / 3600
    else:
        verdict = stage if ok else req["status"]
    req["verdict"] = verdict
    ex.save_state(exp_dir, state)

    if stage == "lr-scan":
        _finish_lr_scan(exp_dir, req, direction)
    elif stage == "bakeoff":
        _finish_bakeoff(exp_dir, req, direction, min_delta)

    desc = req.get("description", "")
    if stage == "research" and not ok:
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
