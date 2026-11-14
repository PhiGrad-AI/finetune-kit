import json
import os
import shutil
import time
from pathlib import Path

import pytest

from finetune_kit import experiment as ex
from finetune_kit import queue, store
from finetune_kit.cli import app
from finetune_kit.paths import kit_home

NOISE = [0.7241, 0.6682, 0.6698]
ENV = {"FINETUNE_KIT_TESTING": "1", "FINETUNE_KIT_FAKE_TRAIN": "1", "FINETUNE_KIT_NO_DAEMON": "1", "FINETUNE_KIT_TEST_PROFILES": "1"}


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("FINETUNE_KIT_TIMEOUT_S", raising=False)
    monkeypatch.chdir(tmp_path)


def _make_exp(tmp_path):
    cfg = ex.ExperimentConfig(
        name="gate", task="seq-cls", model="tiny-bert",
        data=ex.DataPaths(train="train.jsonl"),
        objective=ex.MetricRef(metric="f1"),
        guardrail=ex.MetricRef(metric="accuracy"),
    )
    d = ex.new_experiment(tmp_path, "gate", cfg)
    (d / "splits").mkdir()
    rows = "".join(json.dumps({"id": i, "text": f"t{i}", "label": i % 2}) + "\n" for i in range(20))
    (d / "splits" / "train.jsonl").write_text(rows)
    (d / "splits" / "val.jsonl").write_text(rows)
    mem = tmp_path / ".finetune-kit" / "memory"
    mem.mkdir(parents=True)
    (mem / "research-rules.md").write_text("rules\n")
    ex.commit_space(d, ex.SearchSpace(round=0, tunable=["lr", "dropout"], losses=["ce"]))
    return d


@pytest.fixture
def exp(tmp_path):
    return _make_exp(tmp_path)


def _out(r):
    return json.loads(r.stdout)


def _inject(exp_dir, run_id, fake, **extra):
    p = exp_dir / "runs" / run_id / "request.json"
    req = json.loads(p.read_text())
    req["_fake"] = fake
    req.update(extra)
    p.write_text(json.dumps(req))


def _request(exp_dir, run_id):
    return json.loads((exp_dir / "runs" / run_id / "request.json").read_text())


def _run(runner, exp_dir, config, *args):
    return runner.invoke(app, ["run", exp_dir.name, "--config", json.dumps(config), *args])


def _research(runner, exp_dir, fake, config=None, **extra):
    r = _run(runner, exp_dir, config or {"dropout": 0.2}, "--desc", "try dropout")
    assert r.exit_code == 0, r.stdout
    rid = _out(r)["run_id"]
    _inject(exp_dir, rid, fake, **extra)
    queue.worker_loop()
    return rid


@pytest.fixture(scope="module")
def _noise_template(tmp_path_factory):
    """Noise floor measured once per module (the fake worker still costs ~1.5 s per seed)."""
    from typer.testing import CliRunner

    root = tmp_path_factory.mktemp("noise-template")
    with pytest.MonkeyPatch.context() as mp:
        for k, v in ENV.items():
            mp.setenv(k, v)
        mp.setenv("FINETUNE_KIT_HOME", str(root / "kit-home"))
        mp.delenv("FINETUNE_KIT_TIMEOUT_S", raising=False)
        mp.chdir(root)
        d = _make_exp(root)
        r = _run(CliRunner(), d, {}, "--stage", "noise")
        assert r.exit_code == 0, r.stdout
        _inject(d, _out(r)["run_id"], NOISE, _fake_guard=[0.9, 0.9, 0.9])
        queue.worker_loop()
        assert queue.read_queue() == []
    return d


@pytest.fixture
def exp_with_noise(_noise_template, tmp_path):
    src_root = _noise_template.parent.parent
    for name in ("experiments", ".finetune-kit"):
        shutil.copytree(src_root / name, tmp_path / name)
    return tmp_path / "experiments" / _noise_template.name


def test_research_requires_noise_floor(exp, runner):
    r = _run(runner, exp, {"dropout": 0.2})
    assert r.exit_code == 2 and "noise floor" in _out(r)["error"]


def test_locked_key_rejected(exp_with_noise, runner):
    r = _run(runner, exp_with_noise, {"warmup": 0.1})
    assert r.exit_code == 2 and "warmup" in _out(r)["error"]


def test_string_number_rejected(exp_with_noise, runner):
    r = _run(runner, exp_with_noise, {"lr": "4e-5"})
    assert r.exit_code == 2
    err = _out(r)["error"]
    assert "lr" in err and "number" in err


def test_default_valued_locked_key_accepted(exp_with_noise, runner):
    r = _run(runner, exp_with_noise, {"warmup": 0.05})
    assert r.exit_code == 0, r.stdout
    assert _out(r)["status"] == "queued" and _out(r)["position"] == 1


def test_noise_sets_sigma(exp_with_noise):
    _, _, st = ex.load(exp_with_noise)
    assert st.sigma_obj == pytest.approx(0.0318, abs=1e-4)
    assert st.sigma_guard == 0.0
    assert st.noise_run == st.best_run == "r001"
    assert st.baseline_train_s > 0
    assert st.runs_used == 0 and st.hours_used == 0
    assert _request(exp_with_noise, "r001")["status"] == "done"


def test_noise_locks_model(exp, runner):
    r = _run(runner, exp, {}, "--stage", "noise")
    assert r.exit_code == 0, r.stdout
    assert ex.load(exp)[2].model_locked is True


def test_noise_records_checksums(exp, runner):
    assert set(ex.load(exp)[2].checksums) == {"search-space.yaml"}
    _run(runner, exp, {}, "--stage", "noise")
    assert {"splits/train.jsonl", "splits/val.jsonl", ".finetune-kit/memory/research-rules.md"} <= set(
        ex.load(exp)[2].checksums
    )


def test_noise_uses_default_config_and_scanned_lr(exp, runner):
    st = ex.load(exp)[2]
    st.lr_scan_best = 3e-4
    ex.save_state(exp, st)
    r = _run(runner, exp, {"dropout": 0.3}, "--stage", "noise")
    assert r.exit_code == 0, r.stdout
    assert _request(exp, _out(r)["run_id"])["overrides"] == {"lr": 3e-4}
    assert any("dropout" in w for w in _out(r)["warnings"])


def test_keep_updates_best(exp_with_noise, runner):
    rid = _research(runner, exp_with_noise, [0.75, 0.74, 0.76], _fake_guard=[0.9, 0.9, 0.9])
    req = _request(exp_with_noise, rid)
    assert req["status"] == "done" and req["verdict"] == "keep"
    st = ex.load(exp_with_noise)[2]
    assert st.best_run == rid
    assert st.non_keep_streak == 0 and st.runs_used == 1 and st.hours_used > 0


def test_streak_counts(exp_with_noise, runner):
    for _ in range(2):
        rid = _research(runner, exp_with_noise, [0.68, 0.69, 0.68], _fake_guard=[0.9, 0.9, 0.9])
        assert _request(exp_with_noise, rid)["verdict"] == "discard"
    st = ex.load(exp_with_noise)[2]
    assert st.non_keep_streak == 2 and st.runs_used == 2 and st.best_run == "r001"


def test_crash_is_discard(exp_with_noise, runner):
    r = _run(runner, exp_with_noise, {"dropout": 0.2})
    rid = _out(r)["run_id"]  # no "_fake" -> fake worker raises -> crash
    queue.worker_loop()
    req = _request(exp_with_noise, rid)
    assert req["status"] == "crash" and req["verdict"] == "discard"
    assert (exp_with_noise / "runs" / rid / "seed_0" / "error.txt").is_file()
    st = ex.load(exp_with_noise)[2]
    assert st.non_keep_streak == 1 and st.runs_used == 1


def test_null_objective_is_crash(exp_with_noise, runner):
    rid = _research(runner, exp_with_noise, [None, 0.7, 0.7])
    req = _request(exp_with_noise, rid)
    assert req["status"] == "crash" and req["verdict"] == "discard"
    assert ex.load(exp_with_noise)[2].best_run == "r001"


def test_budget_exhausted(exp_with_noise, runner):
    cfg, _, st = ex.load(exp_with_noise)
    st.runs_used = 1
    ex.save_state(exp_with_noise, st)
    cfg.budget.max_runs = 1
    _write_cfg(exp_with_noise, cfg)
    r = _run(runner, exp_with_noise, {"dropout": 0.2})
    assert r.exit_code == 2 and "budget" in _out(r)["error"]


def test_budget_counts_queued_runs(exp_with_noise, runner):
    cfg, _, _ = ex.load(exp_with_noise)
    cfg.budget.max_runs = 1
    _write_cfg(exp_with_noise, cfg)
    assert _run(runner, exp_with_noise, {"dropout": 0.2}).exit_code == 0
    r = _run(runner, exp_with_noise, {"dropout": 0.3})
    assert r.exit_code == 2 and "budget" in _out(r)["error"]


def test_tamper_blocks_run(exp_with_noise, runner):
    p = exp_with_noise / "search-space.yaml"
    p.write_text(p.read_text().replace("- dropout", "- dropout\n- warmup"))
    r = _run(runner, exp_with_noise, {"warmup": 0.1})
    assert r.exit_code == 2 and "changed" in _out(r)["error"]


def test_stale_running_recovered(exp, runner):
    rid = store.new_run(exp, "noise", {}, [0, 1, 2], "noise")
    _inject(exp, rid, NOISE)
    queue.enqueue(exp, rid)
    req = _request(exp, rid)
    req.update(status="running", pid=999999)
    (exp / "runs" / rid / "request.json").write_text(json.dumps(req))

    recovered = queue.recover_stale()
    assert recovered == [{"exp_dir": str(exp.resolve()), "run_id": rid, "status": "crash", "retried": True}]
    req = _request(exp, rid)
    assert req["status"] == "queued" and req["retries"] == 1
    assert req["attempts"][0]["status"] == "crash"

    queue.worker_loop()
    assert _request(exp, rid)["status"] == "done"
    assert ex.load(exp)[2].sigma_obj == pytest.approx(0.0318, abs=1e-4)
    assert queue.read_queue() == []


def test_stale_running_not_retried_twice(exp):
    rid = store.new_run(exp, "noise", {}, [0, 1, 2], "noise")
    queue.enqueue(exp, rid)
    req = _request(exp, rid)
    req.update(status="running", pid=999999, retries=1)
    (exp / "runs" / rid / "request.json").write_text(json.dumps(req))
    recovered = queue.recover_stale()
    assert recovered[0]["retried"] is False
    assert _request(exp, rid)["status"] == "crash"
    assert queue.read_queue() == []


def test_timeout_marks_timeout(exp_with_noise, runner):
    st = ex.load(exp_with_noise)[2]
    st.baseline_train_s = 0.05
    ex.save_state(exp_with_noise, st)
    rid = _research(runner, exp_with_noise, [0.9, 0.9, 0.9], _fake_sleep=30)
    req = _request(exp_with_noise, rid)
    assert req["status"] == "timeout" and req["verdict"] == "discard"
    assert ex.load(exp_with_noise)[2].non_keep_streak == 1


def test_timeout_env_override(exp_with_noise, runner, monkeypatch):
    monkeypatch.setenv("FINETUNE_KIT_TIMEOUT_S", "0.05")
    rid = _research(runner, exp_with_noise, [0.9, 0.9, 0.9], _fake_sleep=30)
    assert _request(exp_with_noise, rid)["status"] == "timeout"


def test_results_tsv_columns(exp_with_noise, runner):
    rid = _research(runner, exp_with_noise, [0.75, 0.74, 0.76])
    lines = (exp_with_noise / "results.tsv").read_text().splitlines()
    assert lines[0].split("\t") == [
        "run_id", "round", "stage", "objective_mean", "objective_std", "guardrail_mean",
        "n_seeds", "minutes", "verdict", "description", "config",
    ]
    noise_row, row = (l.split("\t") for l in lines[1:])
    assert noise_row[0] == "r001" and noise_row[2] == "noise"
    assert row[0] == rid and row[1] == "1" and row[2] == "research"
    assert float(row[3]) == pytest.approx(0.75)
    assert row[6] == "3" and row[8] == "keep" and row[9] == "try dropout"
    assert json.loads(row[10]) == {"dropout": 0.2}


def test_summarize_counts_nondefault(exp_with_noise, runner):
    rid = _research(runner, exp_with_noise, [0.75, 0.74, 0.76], config={"dropout": 0.2, "lr": 1e-3})
    s = store.summarize(exp_with_noise, rid)
    assert s.n_seeds == 3 and s.n_nondefault == 1  # lr equals tiny-bert's default
    assert s.objective_mean == pytest.approx(0.75)


def test_wait_returns_seeds_and_verdict(exp_with_noise, runner):
    rid = _research(runner, exp_with_noise, [0.75, 0.74, 0.76])
    r = runner.invoke(app, ["wait", rid, "--exp", exp_with_noise.name, "--timeout", "5"])
    out = _out(r)
    assert r.exit_code == 0 and out["status"] == "done" and out["verdict"] == "keep"
    assert [s["objective"] for s in out["seeds"]] == [0.75, 0.74, 0.76]


def test_wait_times_out(exp_with_noise, runner):
    rid = _out(_run(runner, exp_with_noise, {"dropout": 0.2}))["run_id"]
    r = runner.invoke(app, ["wait", rid, "--exp", exp_with_noise.name, "--timeout", "0"])
    out = _out(r)
    assert r.exit_code == 0 and out["status"] == "queued" and "elapsed_s" in out


def test_status_lists_queue(exp_with_noise, runner):
    rid = _out(_run(runner, exp_with_noise, {"dropout": 0.2}))["run_id"]
    out = _out(runner.invoke(app, ["status", exp_with_noise.name]))
    assert [q["run_id"] for q in out["queue"]] == [rid]
    assert out["running"] is None and out["worker_pid"] is None


def test_lr_scan_sets_best(exp, runner):
    r = runner.invoke(app, ["lr-scan", exp.name])
    assert r.exit_code == 0, r.stdout
    out = _out(r)
    assert len(out["run_ids"]) == 5
    assert out["lrs"][0] == pytest.approx(1e-4) and out["lrs"][-1] == pytest.approx(1e-2)
    for i, rid in enumerate(out["run_ids"]):
        req = _request(exp, rid)
        assert req["seeds"] == [0] and req["overrides"]["step_fraction"] == 0.2
        _inject(exp, rid, [[0.5, 0.6, 0.8, 0.7, 0.4][i]])
    queue.worker_loop()
    assert ex.load(exp)[2].lr_scan_best == pytest.approx(out["lrs"][2])


def test_bakeoff_records_ranking(exp, runner):
    r = runner.invoke(app, ["bakeoff", exp.name, "--models", "tiny-bert,tiny-roberta"])
    assert r.exit_code == 0, r.stdout
    ids = _out(r)["run_ids"]
    assert _request(exp, ids[1])["overrides"] == {"model": "tiny-roberta"}
    _inject(exp, ids[0], [0.80, 0.81, 0.80])
    _inject(exp, ids[1], [0.60, 0.61, 0.60])
    queue.worker_loop()
    st = ex.load(exp)[2]
    assert st.bakeoff["winner"] == "tiny-bert"
    out = _out(runner.invoke(app, ["bakeoff", exp.name]))
    assert out["bakeoff"]["winner"] == "tiny-bert"


def test_bakeoff_rejects_unsupported_model(exp, runner):
    r = runner.invoke(app, ["bakeoff", exp.name, "--models", "tiny-bert,nope"])
    assert r.exit_code == 2 and "nope" in _out(r)["error"]


def test_bakeoff_refused_after_lock(exp, runner):
    r = runner.invoke(app, ["bakeoff", "choose", exp.name, "tiny-roberta"])
    assert r.exit_code == 0, r.stdout
    cfg, _, st = ex.load(exp)
    assert cfg.model == "tiny-roberta" and cfg.versions["model_revision"] == "main"
    assert st.model_locked
    r = runner.invoke(app, ["bakeoff", exp.name, "--models", "tiny-bert,tiny-roberta"])
    assert r.exit_code == 2 and "model is locked" in _out(r)["error"]


def test_version_mismatch_warns(exp, runner):
    cfg, _, _ = ex.load(exp)
    cfg.versions = {"torch": "0.0.1"}
    (exp / "experiment.yaml").write_text(__import__("yaml").safe_dump(cfg.model_dump(mode="json")))
    r = _run(runner, exp, {}, "--stage", "noise")
    assert r.exit_code == 0
    assert any("torch" in w for w in _out(r)["warnings"])


def test_enqueue_spawns_daemon_once(exp, monkeypatch):
    monkeypatch.delenv("FINETUNE_KIT_NO_DAEMON")
    calls = []

    class FakeProc:
        pid = os.getpid()  # a live pid

    def fake_popen(cmd, **kw):
        calls.append((cmd, kw))
        return FakeProc()

    monkeypatch.setattr(queue.subprocess, "Popen", fake_popen)
    a = store.new_run(exp, "noise", {}, [0], "a")
    b = store.new_run(exp, "noise", {}, [0], "b")
    assert queue.enqueue(exp, a) == 1
    assert queue.enqueue(exp, b) == 2
    assert len(calls) == 1
    cmd, kw = calls[0]
    assert cmd[1:] == ["-m", "finetune_kit.cli", "_worker"] and kw["start_new_session"] is True
    assert (kit_home() / "worker.lock").read_text().strip() == str(os.getpid())


def test_worker_takes_over_stale_lock(exp):
    kit_home().mkdir(parents=True, exist_ok=True)
    (kit_home() / "worker.lock").write_text("999999")
    rid = store.new_run(exp, "noise", {}, [0, 1, 2], "noise")
    _inject(exp, rid, NOISE)
    queue.enqueue(exp, rid)
    queue.worker_loop()
    assert _request(exp, rid)["status"] == "done"
    assert not (kit_home() / "worker.lock").exists()


def test_worker_defers_to_live_lock(exp):
    kit_home().mkdir(parents=True, exist_ok=True)
    (kit_home() / "worker.lock").write_text(str(os.getppid()))
    rid = store.new_run(exp, "noise", {}, [0, 1, 2], "noise")
    queue.enqueue(exp, rid)
    queue.worker_loop()
    assert _request(exp, rid)["status"] == "queued"


def test_detached_daemon_end_to_end(exp, runner, monkeypatch):
    monkeypatch.delenv("FINETUNE_KIT_NO_DAEMON")
    rid = store.new_run(exp, "noise", {}, [0, 1, 2], "noise")
    _inject(exp, rid, NOISE)
    queue.enqueue(exp, rid)
    r = runner.invoke(app, ["wait", rid, "--exp", exp.name, "--timeout", "60"])
    assert _out(r)["status"] == "done", r.stdout
    deadline = time.time() + 10
    while (kit_home() / "worker.lock").exists() and time.time() < deadline:
        time.sleep(0.1)
    assert not (kit_home() / "worker.lock").exists()  # daemon exited on empty queue


# --- fix round 1 -------------------------------------------------------------------------------


def _write_cfg(exp_dir, cfg):
    """Rewrite experiment.yaml as if it had been set so before the noise floor (checksum re-pinned)."""
    (exp_dir / "experiment.yaml").write_text(__import__("yaml").safe_dump(cfg.model_dump(mode="json")))
    st = ex.load(exp_dir)[2]
    if ex.CONFIG_KEY in st.checksums:
        st.checksums[ex.CONFIG_KEY] = ex._sha(exp_dir / "experiment.yaml")
        ex.save_state(exp_dir, st)


def test_noise_refused_once_done(exp_with_noise, runner):  # 1
    r = _run(runner, exp_with_noise, {}, "--stage", "noise")
    assert r.exit_code == 2 and "new experiment" in _out(r)["error"]


def test_noise_refused_while_pending_allowed_after_crash(exp, runner):  # 1
    rid = _out(_run(runner, exp, {}, "--stage", "noise"))["run_id"]
    r = _run(runner, exp, {}, "--stage", "noise")
    assert r.exit_code == 2 and "new experiment" in _out(r)["error"]
    queue.worker_loop()  # no "_fake" -> crash
    assert _request(exp, rid)["status"] == "crash"
    assert _run(runner, exp, {}, "--stage", "noise").exit_code == 0


def test_lr_scan_refused_after_noise(exp, runner):  # 2
    assert _run(runner, exp, {}, "--stage", "noise").exit_code == 0  # pending is enough
    r = runner.invoke(app, ["lr-scan", exp.name])
    assert r.exit_code == 2 and "noise floor" in _out(r)["error"]


def test_noise_refused_while_lr_scan_pending(exp, runner):  # 2
    assert runner.invoke(app, ["lr-scan", exp.name]).exit_code == 0
    r = _run(runner, exp, {}, "--stage", "noise")
    assert r.exit_code == 2 and "LR scan" in _out(r)["error"]


def test_save_state_is_atomic(exp, monkeypatch):  # 3
    replaced = []
    real = os.replace
    monkeypatch.setattr(ex.os, "replace", lambda a, b: (replaced.append(Path(b).name), real(a, b)))
    ex.save_state(exp, ex.State(round=7))
    assert replaced == ["state.json"] and ex.load(exp)[2].round == 7


def test_failed_verdict_still_counts(exp_with_noise, runner):  # 3
    shutil.rmtree(exp_with_noise / "runs" / "r001" / "seed_0")
    shutil.rmtree(exp_with_noise / "runs" / "r001" / "seed_1")
    shutil.rmtree(exp_with_noise / "runs" / "r001" / "seed_2")  # best run unreadable
    rid = _research(runner, exp_with_noise, [0.75, 0.74, 0.76])
    req = _request(exp_with_noise, rid)
    assert req["verdict"] == "discard" and "verdict failed" in req["error"]
    st = ex.load(exp_with_noise)[2]
    assert st.runs_used == 1 and st.non_keep_streak == 1
    assert rid in store.result_run_ids(exp_with_noise)


def test_parallel_runs_respect_budget(exp_with_noise, tmp_path):  # 4
    import subprocess
    import sys

    cfg = ex.load(exp_with_noise)[0]
    cfg.budget.max_runs = 2
    _write_cfg(exp_with_noise, cfg)
    cmd = [sys.executable, "-m", "finetune_kit.cli", "run", str(exp_with_noise), "--config", '{"dropout": 0.2}']
    procs = [subprocess.Popen(cmd, cwd=tmp_path, stdout=subprocess.PIPE, text=True) for _ in range(6)]
    codes = [p.wait(timeout=60) for p in procs]
    for p in procs:
        p.stdout.close()
    assert sorted(codes) == [0, 0, 2, 2, 2, 2]
    assert len([r for r in store.list_requests(exp_with_noise) if r["stage"] == "research"]) == 2


def test_tamper_after_enqueue_crashes_at_execution(exp_with_noise, runner):  # 5
    rid = _out(_run(runner, exp_with_noise, {"dropout": 0.2}))["run_id"]
    _inject(exp_with_noise, rid, [0.9, 0.9, 0.9])
    (exp_with_noise / "splits" / "val.jsonl").write_text("tampered\n")
    queue.worker_loop()
    req = _request(exp_with_noise, rid)
    assert req["status"] == "crash" and req["verdict"] == "discard"
    assert "refused at execution" in req["error"] and "changed" in req["error"]
    assert not (exp_with_noise / "runs" / rid / "seed_0").exists()
    assert ex.load(exp_with_noise)[2].runs_used == 1


def test_config_revalidated_at_execution(exp_with_noise, runner):  # 5
    rid = _out(_run(runner, exp_with_noise, {"dropout": 0.2}))["run_id"]
    _inject(exp_with_noise, rid, [0.9, 0.9, 0.9])
    ex.commit_space(exp_with_noise, ex.SearchSpace(round=0, tunable=["lr"], losses=["ce"]))
    queue.worker_loop()
    req = _request(exp_with_noise, rid)
    assert req["status"] == "crash" and "dropout" in req["error"]


def test_scanned_lr_not_revalidated_as_agent_key(exp, runner):  # 5
    st = ex.load(exp)[2]
    st.sigma_obj, st.sigma_guard, st.best_run = 0.03, 0.0, None
    st.lr_scan_best = 3e-4
    ex.save_state(exp, st)
    ex.commit_space(exp, ex.SearchSpace(round=0, tunable=["dropout"], losses=["ce"]))  # lr locked
    rid = _out(_run(runner, exp, {"dropout": 0.2}))["run_id"]
    assert _request(exp, rid)["overrides"] == {"lr": 3e-4, "dropout": 0.2}
    _inject(exp, rid, [0.9, 0.9, 0.9])
    queue.worker_loop()
    assert _request(exp, rid)["status"] == "done"


def _mock_popen(monkeypatch):
    calls = []

    class FakeProc:
        pid = os.getpid()

    monkeypatch.setattr(queue.subprocess, "Popen", lambda cmd, **kw: (calls.append(cmd), FakeProc())[1])
    return calls


@pytest.mark.parametrize("command", ["status", "wait"])
def test_dead_daemon_restarted(exp, runner, monkeypatch, command):  # 6
    rid = store.new_run(exp, "noise", {}, [0], "a")
    queue.enqueue(exp, rid)  # NO_DAEMON: nothing spawned
    kit_home().joinpath("worker.lock").write_text("999999")  # daemon died
    monkeypatch.delenv("FINETUNE_KIT_NO_DAEMON")
    calls = _mock_popen(monkeypatch)
    args = ["status"] if command == "status" else ["wait", rid, "--exp", exp.name, "--timeout", "0"]
    out = _out(runner.invoke(app, args))
    assert len(calls) == 1 and calls[0][1:] == ["-m", "finetune_kit.cli", "_worker"]
    if command == "status":
        assert out["worker_pid"] == os.getpid()


def test_status_does_not_spawn_on_empty_queue(runner, monkeypatch):  # 6
    monkeypatch.delenv("FINETUNE_KIT_NO_DAEMON")
    calls = _mock_popen(monkeypatch)
    assert _out(runner.invoke(app, ["status"]))["worker_pid"] is None and calls == []


def test_recover_kills_orphaned_training_group(exp):  # 7
    import subprocess

    child = subprocess.Popen(["sleep", "60"], start_new_session=True)
    rid = store.new_run(exp, "noise", {}, [0, 1, 2], "noise")
    queue.enqueue(exp, rid)
    req = _request(exp, rid)
    req.update(status="running", pid=999999, child_pid=child.pid)
    (exp / "runs" / rid / "request.json").write_text(json.dumps(req))
    queue.recover_stale()
    assert child.wait(timeout=5) == -9
    assert _request(exp, rid)["child_pid"] is None


def test_seed_runs_in_own_session_and_pid_recorded(exp_with_noise, runner, monkeypatch):  # 7
    seen = []
    real = queue.subprocess.Popen

    def spy(cmd, **kw):
        seen.append(kw.get("start_new_session"))
        return real(cmd, **kw)

    monkeypatch.setattr(queue.subprocess, "Popen", spy)
    writes = []
    real_write = store.write_request
    monkeypatch.setattr(store, "write_request", lambda e, r, q: (writes.append(q.get("child_pid")), real_write(e, r, q)))
    _research(runner, exp_with_noise, [0.75, 0.74, 0.76])
    assert seen == [True, True, True]
    assert sum(1 for w in writes if w) == 3


def test_finalize_is_idempotent(exp_with_noise, runner):  # 8
    rid = _research(runner, exp_with_noise, [0.75, 0.74, 0.76])
    before = ex.load(exp_with_noise)[2]
    req = _request(exp_with_noise, rid)
    queue._finalize(exp_with_noise, req, req["seed_status"])
    after = ex.load(exp_with_noise)[2]
    assert after.runs_used == before.runs_used == 1 and after.hours_used == before.hours_used
    rows = (exp_with_noise / "results.tsv").read_text().splitlines()
    assert [r.split("\t")[0] for r in rows[1:]] == ["r001", rid]
    assert _request(exp_with_noise, rid)["verdict"] == "keep"


def test_finalizing_run_recovered_without_retraining(exp_with_noise, runner):  # 8
    rid = _research(runner, exp_with_noise, [0.75, 0.74, 0.76])
    req = _request(exp_with_noise, rid)
    # simulate: worker died after saving state, before the row and terminal request
    lines = (exp_with_noise / "results.tsv").read_text().splitlines()
    (exp_with_noise / "results.tsv").write_text("\n".join(lines[:-1]) + "\n")
    req.update(status="finalizing", outcome="done", pid=999999)
    req.pop("verdict")
    (exp_with_noise / "runs" / rid / "request.json").write_text(json.dumps(req))
    queue.enqueue(exp_with_noise, rid)
    out = queue.recover_stale()
    assert out[0]["retried"] is False
    req = _request(exp_with_noise, rid)
    assert req["status"] == "done" and req["verdict"] == "keep" and req["retries"] == 0
    assert ex.load(exp_with_noise)[2].runs_used == 1
    assert rid in store.result_run_ids(exp_with_noise)
    assert queue.read_queue() == []


def test_test_envs_need_testing_flag(exp, monkeypatch):  # minor 9
    monkeypatch.delenv("FINETUNE_KIT_TESTING")
    monkeypatch.setenv("FINETUNE_KIT_TIMEOUT_S", "0.01")
    st = ex.State(baseline_train_s=10.0)
    assert queue.timeout_for(exp, "research", st) == 30.0
    calls = _mock_popen(monkeypatch)  # FINETUNE_KIT_NO_DAEMON=1 is still set, but ignored
    queue.enqueue(exp, store.new_run(exp, "noise", {}, [0], "a"))
    assert len(calls) == 1


# --- fix round 2 -------------------------------------------------------------------------------


def _edit_overrides(exp_dir, run_id, overrides):
    p = exp_dir / "runs" / run_id / "request.json"
    req = json.loads(p.read_text())
    req["overrides"] = overrides
    p.write_text(json.dumps(req))


def test_edited_research_overrides_caught(exp_with_noise, runner):
    rid = _out(_run(runner, exp_with_noise, {"dropout": 0.2}))["run_id"]
    _inject(exp_with_noise, rid, [0.9, 0.9, 0.9])
    _edit_overrides(exp_with_noise, rid, {"dropout": 0.2, "warmup": 0.3})
    queue.worker_loop()
    req = _request(exp_with_noise, rid)
    assert req["status"] == "crash" and req["verdict"] == "discard"
    assert "request.json was modified after enqueue" in req["error"]
    assert not (exp_with_noise / "runs" / rid / "seed_0").exists()


def test_edited_noise_overrides_caught(exp, runner):
    rid = _out(_run(runner, exp, {}, "--stage", "noise"))["run_id"]
    _inject(exp, rid, NOISE)
    _edit_overrides(exp, rid, {"lr": 5e-4})
    queue.worker_loop()
    req = _request(exp, rid)
    assert req["status"] == "crash" and "modified after enqueue" in req["error"]
    assert ex.load(exp)[2].sigma_obj is None


def test_mark_failed_restores_finalizing_outcome(exp_with_noise, runner):
    rid = _research(runner, exp_with_noise, [0.75, 0.74, 0.76])
    req = _request(exp_with_noise, rid)
    req.update(status="finalizing", outcome="done")
    req.pop("verdict")
    (exp_with_noise / "runs" / rid / "request.json").write_text(json.dumps(req))
    queue._mark_failed(exp_with_noise, rid, "boom")
    req = _request(exp_with_noise, rid)
    assert req["status"] == "done" and req["verdict"] == "keep" and "outcome" not in req
    assert ex.load(exp_with_noise)[2].runs_used == 1


def test_noise_records_experiment_and_profile_checksums(exp, runner):  # final review 5
    assert _run(runner, exp, {}, "--stage", "noise").exit_code == 0
    assert {"experiment.yaml", "model-profile"} <= set(ex.load(exp)[2].checksums)


def test_edited_experiment_yaml_refused_after_noise(exp_with_noise, runner):  # final review 5
    cfg = ex.load(exp_with_noise)[0]
    cfg.budget.max_runs = 999
    (exp_with_noise / "experiment.yaml").write_text(__import__("yaml").safe_dump(cfg.model_dump(mode="json")))
    r = _run(runner, exp_with_noise, {"dropout": 0.2})
    assert r.exit_code == 2 and "experiment.yaml (changed)" in _out(r)["error"]


def test_edited_model_profile_refused_after_noise(exp_with_noise, runner, monkeypatch):  # final review 5
    import dataclasses

    from finetune_kit import profiles

    real = profiles._test_profiles
    monkeypatch.setattr(profiles, "_test_profiles", lambda: {
        k: dataclasses.replace(v, lr_range=(1e-6, 1.0)) for k, v in real().items()
    })
    r = _run(runner, exp_with_noise, {"dropout": 0.2})
    assert r.exit_code == 2 and "model profile 'tiny-bert' (changed)" in _out(r)["error"]
