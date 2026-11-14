import json
import os
import shutil
import time

import pytest

from finetune_kit import experiment as ex
from finetune_kit import queue, store
from finetune_kit.cli import app
from finetune_kit.paths import kit_home

NOISE = [0.7241, 0.6682, 0.6698]
ENV = {"FINETUNE_KIT_FAKE_TRAIN": "1", "FINETUNE_KIT_NO_DAEMON": "1", "FINETUNE_KIT_TEST_PROFILES": "1"}


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
    cfg.budget.max_runs = 1
    (exp_with_noise / "experiment.yaml").write_text(
        __import__("yaml").safe_dump(cfg.model_dump(mode="json"))
    )
    st.runs_used = 1
    ex.save_state(exp_with_noise, st)
    r = _run(runner, exp_with_noise, {"dropout": 0.2})
    assert r.exit_code == 2 and "budget" in _out(r)["error"]


def test_budget_counts_queued_runs(exp_with_noise, runner):
    cfg, _, _ = ex.load(exp_with_noise)
    cfg.budget.max_runs = 1
    (exp_with_noise / "experiment.yaml").write_text(
        __import__("yaml").safe_dump(cfg.model_dump(mode="json"))
    )
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
