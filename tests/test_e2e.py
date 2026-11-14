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
    assert chk  # splits frozen, test moved to the vault
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
    assert "pauc_hi" in json.dumps(test_m)

    rows = tmp_path / "rows.jsonl"
    out = tmp_path / "scores.jsonl"
    rows.write_text("".join(json.dumps(r) + "\n" for r in seq_rows(6)))
    p = subprocess.run(
        [sys.executable, str(final / "predict.py"), "--input", str(rows), "--output", str(out), "--device", "cpu"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert p.returncode == 0, p.stderr
    scores = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(scores) == 6 and all(s for s in scores)


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
    assert "entity_f1" in json.dumps(test_m)
