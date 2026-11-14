import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def _kit_home(tmp_path, monkeypatch):
    home = tmp_path / "kit-home"
    monkeypatch.setenv("FINETUNE_KIT_HOME", str(home))
    return home
