import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def _kit_home(tmp_path, monkeypatch):
    home = tmp_path / "kit-home"
    monkeypatch.setenv("FINETUNE_KIT_HOME", str(home))
    monkeypatch.setenv("FINETUNE_KIT_TESTING", "1")  # enables FAKE_TRAIN, TIMEOUT_S, MAX_STEPS, NO_DAEMON
    return home


def _write_jsonl(path, rows):
    import json

    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.fixture
def seq_jsonl(tmp_path):
    def write(rows, name="data.jsonl"):
        return _write_jsonl(tmp_path / name, rows)

    return write


@pytest.fixture
def tok_jsonl(tmp_path):
    def write(rows, name="data.jsonl"):
        return _write_jsonl(tmp_path / name, rows)

    return write
