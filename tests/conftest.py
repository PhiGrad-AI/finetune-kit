import datasets
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


@pytest.fixture
def hf_fake(monkeypatch):
    """Stand-in for Hugging Face: hf_fake({split: rows}, features=None, sha=...) so tests never use the network."""
    from finetune_kit import sources

    def install(splits, features=None, sha="abc123"):
        feats = features if features is not None else {"label": datasets.ClassLabel(names=["benign", "injection"])}
        monkeypatch.setattr(sources, "_resolve_revision", lambda repo, rev, cfg=None: sha)
        monkeypatch.setattr(
            sources, "_load_dataset", lambda repo, cfg, rev: {k: (v, feats) for k, v in splits.items()}
        )

    return install
