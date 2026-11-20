"""End-to-end smoke: the real CLI, real tiny-model training on CPU, in-process worker."""
import json
import random
import subprocess
import sys
from math import isfinite
from pathlib import Path

import pytest
from typer.testing import CliRunner

from finetune_kit import queue
from finetune_kit.cli import app

pytestmark = [
    pytest.mark.model,
    pytest.mark.filterwarnings("ignore:.*is_gradient_checkpointing.*:FutureWarning"),  # peft
]

WORDS = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta"]
ENT = ["Ann", "Lee", "Kim", "Raj"]


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    for k, v in {
        "FINETUNE_KIT_TESTING": "1", "FINETUNE_KIT_TEST_PROFILES": "1", "FINETUNE_KIT_DEVICE": "cpu",
        "FINETUNE_KIT_MAX_STEPS": "3", "FINETUNE_KIT_NO_DAEMON": "1",
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("FINETUNE_KIT_FAKE_TRAIN", raising=False)
    monkeypatch.delenv("FINETUNE_KIT_TIMEOUT_S", raising=False)
    proj = tmp_path / "project"
    proj.mkdir()
    monkeypatch.chdir(proj)
    return proj


def cli(*args, ok=True) -> dict:
    r = CliRunner().invoke(app, [str(a) for a in args])
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert (r.exit_code == 0) == ok, f"{args}: {r.stdout}"
    return out


def drain(exp: str, *run_ids: str) -> list[dict]:
    """Run the queue in-process, then collect each run's final `wait` payload."""
    queue.worker_loop()
    outs = [cli("wait", rid, "--exp", exp, "--timeout", 30) for rid in run_ids]
    for o in outs:
        assert o["status"] == "done", o
    return outs


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path.name


def seq_rows(n=60) -> list[dict]:
    rng = random.Random(0)
    rows = []
    for i in range(n):
        pos = i % 2 == 1
        words = rng.choices(WORDS, k=4)
        if pos:
            words.insert(rng.randrange(5), rng.choice(["me@x.org", "call 5551234", "id 99"]))
        rows.append({"id": f"r{i}", "text": " ".join(words), "label": int(pos)})
    return rows


def tok_rows(n=40) -> list[dict]:
    rng = random.Random(1)
    rows = []
    for i in range(n):
        toks, tags = rng.choices(WORDS, k=4), ["O"] * 4
        if i % 2:
            toks += [rng.choice(ENT), rng.choice(ENT)]
            tags += ["B-PER", "I-PER"]
        rows.append({"id": f"t{i}", "tokens": toks, "tags": tags})
    return rows


def test_seq_cls_full_flow(_env, tmp_path):
    proj = _env
    data = write_jsonl(proj / "data.jsonl", seq_rows())
    init = CliRunner().invoke(app, ["init", "--ai", "claude", "--here"])  # prints text, not JSON
    assert init.exit_code == 0, init.stdout
    assert (proj / ".finetune-kit" / "memory" / "research-rules.md").is_file()

    name = cli("new", "e2e", "--task", "seq-cls", "--model", "tiny-bert", "--train", data,
               "--metric", "pauc_hi", "--seeds", 2)["experiment"]
    chk = cli("check-data", name)
    assert chk["n_rows"] == {"train": 48, "val": 6, "test": 6} and chk["errors"] == []
    assert set(chk["label_dist"]["train"]) == {"0", "1"} and chk["unseen_labels"] == []
    exp_dir = proj / "experiments" / name
    assert (exp_dir / "splits" / "train.jsonl").is_file() and not (exp_dir / "splits" / "test.jsonl").exists()
    assert list((tmp_path / "kit-home" / "vault").glob("*/test.jsonl"))

    bo = cli("bakeoff", name, "--models", "tiny-bert,tiny-roberta")
    drain(name, *bo["run_ids"])
    result = cli("bakeoff", name)
    assert result["pending"] == 0 and result["bakeoff"]
    assert cli("bakeoff", "choose", name, "tiny-bert")["model_locked"] is True

    assert cli("space", "commit", name, "--tunable", "lr,llrd", "--losses", "ce")["round"] == 1

    scan = cli("lr-scan", name)
    drain(name, *scan["run_ids"])

    noise = cli("run", name, "--stage", "noise")
    n_out = drain(name, noise["run_id"])[0]
    assert n_out["verdict"] == "baseline"

    research = cli("run", name, "--config", '{"llrd":0.8}', "--desc", "llrd 0.8")
    r_out = drain(name, research["run_id"])[0]
    assert r_out["verdict"] in ("keep", "keep-simpler", "discard")
    for seed in r_out["seeds"]:
        assert isfinite(seed["objective"]) and all(isfinite(v) for v in seed["report"].values()), seed

    cmp = cli("compare", name)
    assert cmp["best"]
    by_stage = {}
    for r in cmp["runs"]:
        by_stage.setdefault(r["stage"], []).append(r)
    assert {k: len(v) for k, v in by_stage.items()} == {"bakeoff": 2, "lr-scan": 5, "noise": 1, "research": 1}
    assert by_stage["noise"][0]["run_id"] == noise["run_id"]
    assert by_stage["research"][0]["run_id"] == research["run_id"] and by_stage["research"][0]["verdict"]

    html = Path(cli("report", name)["path"])
    assert html.is_file() and html.stat().st_size > 0

    best = cmp["best"]
    miss = cli("misses", name, best)
    assert "false_negatives" in miss

    pr = cli("promote", name, best, "--seeds", 1)
    drain(name, pr["run_id"])
    final = exp_dir / "final"
    for f in ("recipe.json", "test_metrics.json", "throughput.json", "predict.py"):
        assert (final / f).is_file(), f
    test_m = json.loads((final / "test_metrics.json").read_text())
    assert test_m["headline"]["metric"] == "pauc_hi" and isfinite(test_m["headline"]["value"])
    assert isfinite(test_m["report"]["pauc_hi"]) and test_m["n_rows"] == 6

    rows = tmp_path / "rows.jsonl"
    out = tmp_path / "scores.jsonl"
    rows.write_text("".join(json.dumps(r) + "\n" for r in seq_rows(6)))
    p = subprocess.run(
        [sys.executable, str(final / "predict.py"), "--input", str(rows), "--output", str(out), "--device", "cpu"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert p.returncode == 0, p.stderr
    scores = [json.loads(l) for l in out.read_text().splitlines()]
    assert [s["id"] for s in scores] == [f"r{i}" for i in range(6)]
    for s in scores:
        assert isinstance(s["score"], float) and isfinite(s["score"]) and 0.0 <= s["score"] <= 1.0, s
        assert s["label"] in ("0", "1"), s


def test_token_cls_flow(_env, tmp_path):
    proj = _env
    data = write_jsonl(proj / "data.jsonl", tok_rows())
    assert CliRunner().invoke(app, ["init", "--ai", "claude", "--here"]).exit_code == 0
    name = cli("new", "ner", "--task", "token-cls", "--model", "tiny-bert", "--train", data,
               "--metric", "entity_f1", "--seeds", 2)["experiment"]
    cli("check-data", name)
    cli("space", "commit", name, "--tunable", "lr,llrd", "--losses", "ce")
    noise = cli("run", name, "--stage", "noise")
    n_out = drain(name, noise["run_id"])[0]
    for seed in n_out["seeds"]:
        assert isfinite(seed["objective"]), seed
    cmp = cli("compare", name)
    assert len(cmp["runs"]) == 1
    pr = cli("promote", name, noise["run_id"], "--seeds", 1)
    drain(name, pr["run_id"])
    test_m = json.loads((proj / "experiments" / name / "final" / "test_metrics.json").read_text())
    assert test_m["objective_metric"] == "entity_f1" and isfinite(test_m["objective"])
    assert test_m["report"]["entity_f1"] == test_m["objective"]


def _noise_run_ok(name: str) -> None:
    noise = cli("run", name, "--stage", "noise")
    out = drain(name, noise["run_id"])[0]
    assert out["verdict"] == "baseline"
    assert all(isfinite(s["objective"]) for s in out["seeds"]), out


def test_e2e_csv_source(_env):
    proj = _env
    rows = seq_rows()
    lines = ["row,prompt,is_injection"] + [f'{r["id"]},"{r["text"]}",{r["label"]}' for r in rows]
    (proj / "data.csv").write_text("\n".join(lines) + "\n")
    cols = '{"text":"prompt","label":"is_injection"}'

    insp = cli("inspect-data", "--train", "data.csv")
    assert isinstance(insp, dict)

    name = cli("new", "csvexp", "--task", "seq-cls", "--model", "tiny-bert", "--train", "data.csv",
               "--columns", cols, "--metric", "pauc_hi", "--seeds", 2)["experiment"]
    chk = cli("check-data", name)
    assert chk["errors"] == [] and set(chk["label_dist"]["train"]) == {"0", "1"}
    _noise_run_ok(name)


def test_e2e_hf_source(_env, hf_fake):
    rows = [{"text": r["text"], "label": r["label"]} for r in seq_rows(72)]
    hf_fake({"train": rows[:48], "validation": rows[48:60], "test": rows[60:]})

    insp = cli("inspect-data", "--source", "hf:fake/ds")
    assert isinstance(insp, dict)

    name = cli("new", "hfexp", "--task", "seq-cls", "--model", "tiny-bert", "--source", "hf:fake/ds",
               "--metric", "pauc_hi", "--seeds", 2)["experiment"]
    chk = cli("check-data", name)
    assert chk["errors"] == []
    exp_dir = _env / "experiments" / name
    frozen = [json.loads(l) for l in (exp_dir / "splits" / "train.jsonl").read_text().splitlines()]
    assert {r["label"] for r in frozen} == {"benign", "injection"}
    report = json.loads((exp_dir / "data-report.json").read_text())
    assert set(report["split_origin"].values()) == {"published"}
    _noise_run_ok(name)


def test_e2e_guided_steps(_env):
    """Drive the CLI the way the step prompts do, asserting `next` after every answer, then a real round 2."""
    from finetune_kit import experiment as ex

    proj = _env
    write_jsonl(proj / "data.jsonl", seq_rows())
    assert CliRunner().invoke(app, ["init", "--ai", "claude", "--here"]).exit_code == 0

    assert cli("next")["stage"] == "choose-task"
    name = cli("new", "guided", "--task", "seq-cls")["experiment"]
    exp_dir = proj / "experiments" / name
    assert cli("next")["stage"] == "choose-data"
    cli("set", "data", name, "--train", "data.jsonl")
    assert cli("next")["stage"] == "choose-model"
    cli("set", "model", name, "tiny-bert")
    assert cli("next")["stage"] == "choose-goal"
    cli("set", "goal", name, "--metric", "pauc_hi")
    assert cli("next")["stage"] == "check-data"
    assert cli("check-data", name)["errors"] == []
    assert cli("next")["stage"] == "choose-loss"
    cli("set", "losses", name, "ce")
    assert cli("next")["stage"] == "choose-hyperparameters"
    assert cli("space", "commit", name, "--tunable", "lr,llrd")["round"] == 1
    out = cli("next")
    assert out["stage"] == "choose-learning-rate"
    cli("set", "learning-rate", name, "default")
    assert cli("next")["stage"] == "choose-experiments"
    cli("set", "experiments", name, "--seeds", 2, "--max-runs", 4)
    out = cli("next")
    assert out["stage"] == "start-experiment" and "hasn't run yet" in out["reason"]

    _noise_run_ok(name)
    out = cli("next")
    assert out["stage"] == "start-experiment" and "still running" in out["reason"]
    steps = cli("steps")
    status = {s["step"]: s["status"] for s in steps["steps"]}
    assert status["choose-experiments"] == "done" and status["start-experiment"] == "next"

    # the research loop's own files, as the start-experiment and report prompts write them
    (exp_dir / "log.md").write_text("## Summary (round 1)\nbaseline only\n")
    assert cli("next")["stage"] == "report"
    (exp_dir / "report.md").write_text("# report\n")
    assert cli("next")["stage"] == "finish"

    # round 2: new losses, new tunables, no re-asking of the once-only steps
    cli("set", "losses", name, "ce,focal")
    assert cli("next")["stage"] == "choose-hyperparameters"
    assert cli("space", "commit", name, "--tunable", "lr,llrd,weight_decay")["round"] == 2
    out = cli("next")
    assert out["stage"] == "start-experiment" and "round 2" in out["reason"]
    status = {s["step"]: s["status"] for s in cli("steps")["steps"]}
    assert status["choose-learning-rate"] == "done" and status["choose-experiments"] == "done"
    assert status["start-experiment"] == "next"
    ex.verify_checksums(exp_dir, proj)
