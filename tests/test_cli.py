import json

import pytest

import finetune_kit
from finetune_kit.cli import app
from finetune_kit.paths import find_experiment, kit_home, vault_dir


def test_version_prints_json(runner):
    r = runner.invoke(app, ["version"])
    assert r.exit_code == 0 and json.loads(r.stdout)["version"] == finetune_kit.__version__


def test_find_experiment_by_number(tmp_path):
    (tmp_path / "experiments/001-pii").mkdir(parents=True)
    assert find_experiment("001", tmp_path).name == "001-pii"


def test_find_experiment_full_name_and_path(tmp_path):
    d = tmp_path / "experiments/001-pii"
    d.mkdir(parents=True)
    assert find_experiment("001-pii", tmp_path) == d
    assert find_experiment(str(d), tmp_path) == d


def test_find_experiment_bare_name(tmp_path):
    (tmp_path / "experiments/001-e2e").mkdir(parents=True)
    assert find_experiment("e2e", tmp_path).name == "001-e2e"


def test_find_experiment_ambiguous(tmp_path):
    (tmp_path / "experiments/001-e2e").mkdir(parents=True)
    (tmp_path / "experiments/002-e2e").mkdir(parents=True)
    with pytest.raises(ValueError, match="001-e2e.*002-e2e"):
        find_experiment("e2e", tmp_path)


def test_find_experiment_missing_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_experiment("009", tmp_path)


def test_kit_home_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FINETUNE_KIT_HOME", str(tmp_path))
    assert kit_home() == tmp_path
    assert vault_dir("abc") == tmp_path / "vault" / "abc"


def test_fail_emits_error_json(runner):
    import typer

    from finetune_kit import cli

    t = typer.Typer()

    @t.command()
    def boom():
        cli.fail("nope")

    @t.command()
    def other():
        pass

    r = runner.invoke(t, ["boom"])
    assert r.exit_code == 2 and json.loads(r.stdout) == {"error": "nope"}


def test_init_force_help_warns_about_tamper_check(runner):
    r = runner.invoke(app, ["init", "--help"], env={"COLUMNS": "200"})
    assert "tamper check" in " ".join(r.stdout.split())
