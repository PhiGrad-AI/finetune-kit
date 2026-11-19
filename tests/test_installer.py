import json
from importlib import resources

import pytest
import yaml

from finetune_kit import guard, installer
from finetune_kit.cli import app

STAGES = ["setup", "check", "bakeoff", "hparams", "baseline", "research", "analyze", "promote"]
GUARD = "finetune-kit _guard"


def _guard_hooks(settings: dict) -> list[dict]:
    return [
        h for entry in settings.get("hooks", {}).get("PreToolUse", [])
        for h in entry.get("hooks", []) if h.get("command") == GUARD
    ]


def _frontmatter(text: str) -> dict:
    assert text.startswith("---\n")
    return yaml.safe_load(text.split("---\n")[1])


# ---- packaging ----


def test_templates_ship_as_package_data():
    root = resources.files("finetune_kit") / "templates"
    for stage in STAGES:
        assert (root / "commands" / f"{stage}.md").read_text().strip()
    for doc in ("experiment", "search-space", "baseline", "report"):
        assert (root / "docs" / f"{doc}.md").read_text().strip()
    assert "Keep" in (root / "memory" / "research-rules.md").read_text()
    assert (root / "predict.py.tmpl").is_file()


# ---- claude ----


def test_init_claude_writes_skills_and_hook(tmp_path):
    written = installer.install(tmp_path, "claude", force=False)
    for stage in STAGES:
        skill = tmp_path / ".claude/skills" / f"finetune-kit-{stage}" / "SKILL.md"
        assert skill in written
        fm = _frontmatter(skill.read_text())
        assert fm["name"] == f"finetune-kit-{stage}"
        assert isinstance(fm["description"], str) and "\n" not in fm["description"]
        assert fm["description"].startswith(installer.DESCRIPTIONS[stage][:20])
    assert (tmp_path / ".finetune-kit/memory/research-rules.md").is_file()
    assert len(list((tmp_path / ".finetune-kit/templates").glob("*.md"))) == 4
    assert (tmp_path / "experiments/.gitkeep").is_file()
    settings = json.loads((tmp_path / ".claude/settings.json").read_text())
    entry = settings["hooks"]["PreToolUse"][0]
    assert entry == {"matcher": "Edit|Write|MultiEdit|NotebookEdit", "hooks": [{"type": "command", "command": GUARD}]}


def test_claude_skills_use_claude_invocation(tmp_path):
    installer.install(tmp_path, "claude", force=False)
    text = (tmp_path / ".claude/skills/finetune-kit-setup/SKILL.md").read_text()
    assert "/finetune-kit-<stage>" in text
    assert "$finetune-kit-" not in text and "/finetune-kit." not in text


def test_init_merges_existing_settings(tmp_path):
    """Review focus 4: user settings, hooks and skills survive two inits; our hook appears once."""
    claude = tmp_path / ".claude"
    (claude / "skills/my-skill").mkdir(parents=True)
    (claude / "skills/my-skill/SKILL.md").write_text("---\nname: my-skill\ndescription: mine\n---\nhi\n")
    user_hook = {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo hi"}]}
    (claude / "settings.json").write_text(json.dumps({
        "permissions": {"allow": ["Bash(ls:*)"]},
        "hooks": {"PreToolUse": [user_hook], "Stop": [{"hooks": [{"type": "command", "command": "say done"}]}]},
    }))

    installer.install(tmp_path, "claude", force=False)
    installer.install(tmp_path, "claude", force=False)

    settings = json.loads((claude / "settings.json").read_text())
    assert settings["permissions"] == {"allow": ["Bash(ls:*)"]}
    assert settings["hooks"]["Stop"] == [{"hooks": [{"type": "command", "command": "say done"}]}]
    assert settings["hooks"]["PreToolUse"][0] == user_hook
    assert len(_guard_hooks(settings)) == 1
    assert (claude / "skills/my-skill/SKILL.md").read_text().startswith("---\nname: my-skill")


def test_init_rejects_invalid_settings_json(tmp_path):
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude/settings.json").write_text("{not json")
    with pytest.raises(ValueError, match="settings.json"):
        installer.install(tmp_path, "claude", force=False)
    assert (tmp_path / ".claude/settings.json").read_text() == "{not json"


def test_init_keeps_user_files_and_experiments(tmp_path):
    (tmp_path / "experiments/001-pii").mkdir(parents=True)
    (tmp_path / "experiments/001-pii/log.md").write_text("my log")
    (tmp_path / "README.md").write_text("mine")
    (tmp_path / "experiments/.gitkeep").write_text("keep me")
    installer.install(tmp_path, "claude", force=False)
    installer.install(tmp_path, "claude", force=False)
    assert (tmp_path / "experiments/001-pii/log.md").read_text() == "my log"
    assert (tmp_path / "README.md").read_text() == "mine"
    assert (tmp_path / "experiments/.gitkeep").read_text() == "keep me"


def test_init_keeps_modified_rules_unless_forced(tmp_path):
    installer.install(tmp_path, "claude", force=False)
    rules = tmp_path / ".finetune-kit/memory/research-rules.md"
    rules.write_text("my own rules")
    written = installer.install(tmp_path, "claude", force=False)
    assert rules.read_text() == "my own rules" and rules not in written
    written = installer.install(tmp_path, "claude", force=True)
    assert rules.read_text() != "my own rules" and rules in written


def test_init_overwrites_stale_kit_files(tmp_path):
    installer.install(tmp_path, "claude", force=False)
    skill = tmp_path / ".claude/skills/finetune-kit-research/SKILL.md"
    skill.write_text("stale")
    installer.install(tmp_path, "claude", force=False)
    assert skill.read_text() != "stale"


# ---- codex and cursor ----


def test_init_codex_and_cursor_paths(tmp_path):
    installer.install(tmp_path, "codex", force=False)
    installer.install(tmp_path, "cursor", force=False)
    for stage in STAGES:
        skill = tmp_path / ".agents/skills" / f"finetune-kit-{stage}" / "SKILL.md"
        assert _frontmatter(skill.read_text())["name"] == f"finetune-kit-{stage}"
        assert (tmp_path / ".cursor/commands" / f"finetune-kit.{stage}.md").is_file()
    codex = (tmp_path / ".agents/skills/finetune-kit-setup/SKILL.md").read_text()
    assert "$finetune-kit-<stage>" in codex
    cursor = (tmp_path / ".cursor/commands/finetune-kit.setup.md").read_text()
    assert "/finetune-kit.<stage>" in cursor
    mdc = (tmp_path / ".cursor/rules/finetune-kit.mdc").read_text()
    assert "alwaysApply: true" in mdc and ".finetune-kit/memory/research-rules.md" in mdc
    agents = (tmp_path / "AGENTS.md").read_text()
    assert ".finetune-kit/memory/research-rules.md" in agents
    assert not (tmp_path / ".claude").exists()


def test_agents_md_section_replaced_not_duplicated(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# My project\n\nBe nice.\n")
    installer.install(tmp_path, "codex", force=False)
    agents = tmp_path / "AGENTS.md"
    text = agents.read_text()
    agents.write_text(text.replace("research-rules.md", "research-rules.md OLD") + "\n## After\n\nmine\n")
    installer.install(tmp_path, "codex", force=False)
    text = agents.read_text()
    assert text.startswith("# My project\n\nBe nice.\n")
    assert text.count(installer.START) == 1 and text.count(installer.END) == 1
    assert "OLD" not in text and "## After\n\nmine" in text


def test_templates_have_no_unfilled_placeholders(tmp_path):
    written = []
    for ai in ("claude", "codex", "cursor"):
        written += installer.install(tmp_path / ai, ai, force=False)
    for p in written:
        if p.is_file():
            assert "{{" not in p.read_text(), p


def test_unknown_agent_rejected(tmp_path):
    with pytest.raises(ValueError, match="claude, codex, cursor"):
        installer.install(tmp_path, "vim", force=False)


def test_cli_init_here(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["init", "--ai", "claude", "--here"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / ".claude/skills/finetune-kit-setup/SKILL.md").is_file()
    assert "/finetune-kit-setup" in r.output


def test_cli_init_new_project_dir(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["init", "proj", "--ai", "cursor"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "proj/.cursor/commands/finetune-kit.setup.md").is_file()


def test_cli_init_bad_agent_is_json_error(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["init", "--ai", "emacs"])
    assert r.exit_code == 2 and "error" in json.loads(r.stdout)


# ---- guard ----


def _hook(runner, tmp_path, path, tool="Edit", key="file_path"):
    payload = {"cwd": str(tmp_path), "hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": {key: path}}
    return runner.invoke(app, ["_guard"], input=json.dumps(payload))


def test_guard_blocks_splits(runner, tmp_path):
    r = _hook(runner, tmp_path, "experiments/001-x/splits/train.jsonl")
    assert r.exit_code == 2
    assert "protected" in r.stderr


def test_guard_allows_log_md(runner, tmp_path):
    r = _hook(runner, tmp_path, "experiments/001-x/log.md")
    assert r.exit_code == 0 and r.stderr == ""


@pytest.mark.parametrize("path", [
    ".finetune-kit/memory/research-rules.md",
    "experiments/001-x/final/predict.py",
    "experiments/001-x/runs/r001/seed_0/metrics.json",
    "experiments/001-x/search-space.yaml",
    "experiments/001-x/state.json",
    "experiments/001-x/results.tsv",
    "experiments/001-x/experiment.yaml",
    "experiments/001-x/data-report.json",
    "sub/../experiments/001-x/state.json",
])
def test_guard_blocks_protected(tmp_path, path):
    assert guard.blocked(path, str(tmp_path)) is not None


@pytest.mark.parametrize("path", [
    "experiments/001-x/experiment.md",
    "experiments/001-x/report.md",
    "experiments/001-x/notes/state.json",
    ".finetune-kit/templates/report.md",
    "src/runs/x.py",
    "train.py",
])
def test_guard_allows_unprotected(tmp_path, path):
    assert guard.blocked(path, str(tmp_path)) is None


def test_guard_blocks_absolute_path(tmp_path):
    assert guard.blocked(str(tmp_path / "experiments/002-y/runs/r1/x"), "/") is not None


def test_guard_blocks_kit_home(tmp_path, _kit_home):
    assert guard.blocked(str(_kit_home / "vault/abc/test.jsonl"), str(tmp_path)) is not None
    assert guard.blocked("~/.finetune-kit/vault/abc/test.jsonl", str(tmp_path)) is not None


def test_guard_notebook_path(runner, tmp_path):
    r = _hook(runner, tmp_path, "experiments/001-x/runs/a.ipynb", tool="NotebookEdit", key="notebook_path")
    assert r.exit_code == 2


def test_guard_ignores_missing_path_and_bad_json(runner):
    assert runner.invoke(app, ["_guard"], input=json.dumps({"tool_input": {}})).exit_code == 0
    assert runner.invoke(app, ["_guard"], input="not json").exit_code == 0


# ---- prompt content ----

PREVIOUS = {"check": "setup", "bakeoff": "check", "hparams": "check", "baseline": "hparams",
            "research": "baseline", "analyze": "research", "promote": "analyze"}


@pytest.mark.parametrize("stage", STAGES)
def test_command_prompts_follow_the_rules(tmp_path, stage):
    installer.install(tmp_path, "claude", force=False)
    text = (tmp_path / ".claude/skills" / f"finetune-kit-{stage}" / "SKILL.md").read_text()
    assert ".finetune-kit/memory/research-rules.md" in text
    assert "one question at a time" in text
    assert "--timeout 540" in text
    assert '{"error"' in text
    assert "~/.finetune-kit/" in text and "experiments/*/splits/" in text
    if stage in PREVIOUS:
        before = text.split("## Before you start", 1)[1].split("## How to work", 1)[0]
        assert "finetune-kit next" in before and "Shall I start" in before
    finish = text.split("## Finish", 1)[1]
    assert "finetune-kit next" in finish and "Shall I start" in finish


@pytest.mark.parametrize("ai,path", [
    ("claude", ".claude/skills/finetune-kit-setup/SKILL.md"),
    ("codex", ".agents/skills/finetune-kit-setup/SKILL.md"),
    ("cursor", ".cursor/commands/finetune-kit.setup.md"),
])
def test_setup_prompt_mentions_data_sources(tmp_path, ai, path):
    installer.install(tmp_path, ai, force=False)
    text = (tmp_path / path).read_text()
    assert "inspect-data" in text and "--source hf:" in text


def test_research_and_promote_specifics(tmp_path):
    installer.install(tmp_path, "claude", force=False)
    skills = tmp_path / ".claude/skills"
    research = (skills / "finetune-kit-research/SKILL.md").read_text()
    assert "Never ask the human whether to continue" in research
    assert "log.md" in research and "finetune-kit compare" in research
    promote = (skills / "finetune-kit-promote/SKILL.md").read_text()
    assert "exactly once" in promote and "answer yes" in promote


def test_rules_have_keep_formula_and_stop_rule():
    rules = (resources.files("finetune_kit") / "templates/memory/research-rules.md").read_text()
    assert "noise = σ · sqrt(2/N)" in rules
    assert "Δ > max(2·noise, min_delta)" in rules
    assert "σ_guard · sqrt(2/N)" in rules
    assert "|Δ| ≤ 2·noise" in rules
    assert "10 non-keeps in a row" in rules


def test_cursor_command_frontmatter_parses(tmp_path):
    installer.install(tmp_path, "cursor", force=False)
    fm = _frontmatter((tmp_path / ".cursor/commands/finetune-kit.research.md").read_text())
    assert fm["description"] == installer.DESCRIPTIONS["research"]


@pytest.mark.parametrize("settings", [{"hooks": []}, {"hooks": {"PreToolUse": {}}}, {"hooks": {"PreToolUse": [{"hooks": 5}]}}])
def test_init_rejects_malformed_settings_shapes(tmp_path, settings):
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude/settings.json").write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="settings.json"):
        installer.install(tmp_path, "claude", force=False)


# ---- stage hand-off ----

_PREFIX = {"claude": "/finetune-kit-", "codex": "$finetune-kit-", "cursor": "/finetune-kit."}
_STAGE_FILE = {
    "claude": ".claude/skills/finetune-kit-<stage>/SKILL.md",
    "codex": ".agents/skills/finetune-kit-<stage>/SKILL.md",
    "cursor": ".cursor/commands/finetune-kit.<stage>.md",
}


@pytest.mark.parametrize("ai", ["claude", "codex", "cursor"])
@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_ends_with_the_hand_off(stage, ai):
    text = installer.render(stage, ai)
    assert "{{" not in text
    assert "finetune-kit next" in text
    assert "Shall I start" in text
    assert _PREFIX[ai] in text
    assert _STAGE_FILE[ai] in text


@pytest.mark.parametrize("ai", ["claude", "codex", "cursor"])
def test_stage_file_matches_what_install_writes(tmp_path, ai):
    written = {str(p.relative_to(tmp_path)) for p in installer.install(tmp_path, ai, force=False)}
    for stage in STAGES:
        assert _STAGE_FILE[ai].replace("<stage>", stage) in written
    assert not any("_finish" in w or "_goto" in w for w in written)


def test_research_summary_names_its_round():
    assert "## Summary (round" in installer.render("research", "codex")


def test_setup_passes_bakeoff_to_new():
    assert "--bakeoff" in installer.render("setup", "codex")


def test_bakeoff_reads_the_cli_not_prose():
    text = installer.render("bakeoff", "codex")
    assert "the `bakeoff` field of `experiment.yaml`" in text
    assert "Late bake-off" in text and "bakeoff choose <exp> <model>" in text
    assert "prose" not in text and "experiment.md`. If the user did **not**" not in text


def test_finish_says_what_to_do_on_no_and_requires_a_yes():
    text = installer.render("check", "codex").split("## Finish", 1)[1]
    assert "If the user says no, stop" in text
    assert "Never start a stage the user has not agreed to" in text


def test_goto_handles_a_missing_experiment():
    assert "no argument" in installer.render("check", "codex").split("## How to work", 1)[0]


def test_rules_forbid_skipping_a_stage():
    rules = _template_text("memory", "research-rules.md").lower()
    assert "never skip a stage" in rules


def _template_text(*parts):
    return (resources.files("finetune_kit") / "templates").joinpath(*parts).read_text(encoding="utf-8")
