import json
from importlib import resources

import pytest
import yaml

from finetune_kit import guard, installer
from finetune_kit.cli import app

STAGES = ["start", "choose-task", "choose-data", "choose-model", "choose-goal", "check-data", "compare-models",
          "choose-loss", "choose-hyperparameters", "choose-learning-rate", "choose-experiments",
          "start-experiment", "status", "report", "finish"]
OLD_STAGES = ["setup", "check", "bakeoff", "hparams", "baseline", "research", "analyze", "promote"]
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
    text = (tmp_path / ".claude/skills/finetune-kit-start/SKILL.md").read_text()
    assert "/finetune-kit-<step>" in text
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
    skill = tmp_path / ".claude/skills/finetune-kit-start-experiment/SKILL.md"
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
    codex = (tmp_path / ".agents/skills/finetune-kit-start/SKILL.md").read_text()
    assert "$finetune-kit-<step>" in codex
    cursor = (tmp_path / ".cursor/commands/finetune-kit.start.md").read_text()
    assert "/finetune-kit.<step>" in cursor
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
    assert (tmp_path / ".claude/skills/finetune-kit-start/SKILL.md").is_file()
    assert "/finetune-kit-start" in r.output


def test_cli_init_new_project_dir(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["init", "proj", "--ai", "cursor"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "proj/.cursor/commands/finetune-kit.start.md").is_file()


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

GATED = [s for s in STAGES if s not in ("start", "status")]
AIS = ["claude", "codex", "cursor"]
_PREFIX = {"claude": "/finetune-kit-", "codex": "$finetune-kit-", "cursor": "/finetune-kit."}
_STEP_FILE = {
    "claude": ".claude/skills/finetune-kit-<step>/SKILL.md",
    "codex": ".agents/skills/finetune-kit-<step>/SKILL.md",
    "cursor": ".cursor/commands/finetune-kit.<step>.md",
}


def test_stages_match_the_cli_steps():
    from finetune_kit.commands import STEPS
    assert installer.STAGES == [s["step"] for s in STEPS] == STAGES


@pytest.mark.parametrize("stage", STAGES)
def test_command_prompts_follow_the_rules(tmp_path, stage):
    installer.install(tmp_path, "claude", force=False)
    text = (tmp_path / ".claude/skills" / f"finetune-kit-{stage}" / "SKILL.md").read_text()
    assert ".finetune-kit/memory/research-rules.md" in text
    assert "one question at a time" in text
    assert "--timeout 540" in text
    assert '{"error"' in text
    assert "~/.finetune-kit/" in text and "experiments/*/splits/" in text and "~/.cache/huggingface/" in text
    if stage in GATED:
        before = text.split("## Before you start", 1)[1].split("## How to work", 1)[0]
        assert "finetune-kit next" in before and "Shall I start" in before
    else:
        assert "{{ORDER_CHECK}}" not in text and "Check the step order first" not in text
    finish = text.split("## Finish", 1)[1]
    assert "finetune-kit next" in finish and "Shall I start" in finish
    assert "Step <number> of 14" in finish


# Each step's own question (or command), and the questions that belong to other steps only.
OWN = {
    "start": ["finetune-kit steps", "Here are the steps:"],
    "choose-task": ["Which of these fits what you want the model to do?", "finetune-kit new <name> --task"],
    "choose-data": ["Where does your data come from?", "finetune-kit set data <exp>", "inspect-data"],
    "choose-model": ["Which model shall we start from?", "finetune-kit set model <exp>", "--compare"],
    "choose-goal": ["Which score should the model improve?", "finetune-kit set goal <exp>",
                    "Is there a second score that must not get worse"],
    "check-data": ["finetune-kit check-data <exp>"],
    "compare-models": ["finetune-kit bakeoff choose <exp> <model>"],
    "choose-loss": ["Which losses may research try?", "finetune-kit set losses <exp>"],
    "choose-hyperparameters": ["Which of these may I change?", "finetune-kit space commit <exp> --tunable"],
    "choose-learning-rate": ["Shall I use the model's usual learning speed, or run a short test",
                             "finetune-kit set learning-rate <exp>", "finetune-kit lr-scan <exp>"],
    "choose-experiments": ["How many experiments may research run", "finetune-kit set experiments <exp>"],
    "start-experiment": ["finetune-kit run <exp> --stage noise", "Never ask the human whether to continue"],
    "status": ["finetune-kit status <exp>", "finetune-kit compare <exp>", "finetune-kit steps <exp>"],
    "report": ["finetune-kit misses <exp>", "finetune-kit report <exp>"],
    "finish": ["finetune-kit promote <exp>", "exactly once", "answer yes"],
}
QUESTIONS = {
    "choose-task": "Which of these fits what you want the model to do?",
    "choose-data": "Where does your data come from?",
    "choose-model": "Which model shall we start from?",
    "choose-goal": "Which score should the model improve?",
    "choose-loss": "Which losses may research try?",
    "choose-hyperparameters": "Which of these may I change?",
    "choose-learning-rate": "Shall I use the model's usual learning speed",
    "choose-experiments": "How many experiments may research run",
    "finish": "score the test set once? Please answer yes",
}


@pytest.mark.parametrize("ai", AIS)
@pytest.mark.parametrize("stage", STAGES)
def test_each_step_asks_only_its_own_question(stage, ai):
    text = installer.render(stage, ai)
    assert "{{" not in text
    for needle in OWN[stage]:
        assert needle in text, needle
    for other, question in QUESTIONS.items():
        if other != stage:
            assert question not in text, (other, question)


@pytest.mark.parametrize("ai", AIS)
def test_start_lists_every_step_and_offers_the_next(ai):
    text = installer.render("start", ai)
    assert "finetune-kit steps" in text and "Here are the steps:" in text
    for field in ("`number`", "`title`", "`what`"):
        assert field in text
    assert f"{_PREFIX[ai]}choose-task" in text
    assert "Step <number> of 14" in text


@pytest.mark.parametrize("ai", AIS)
@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_ends_with_the_hand_off(stage, ai):
    text = installer.render(stage, ai)
    assert "{{" not in text
    finish = text.split("## Finish", 1)[1]
    assert "finetune-kit next" in finish and "finetune-kit steps" in finish
    assert "Shall I start" in finish and "Step <number> of 14" in finish
    assert f"Command: {_PREFIX[ai]}<step>" in finish
    assert _STEP_FILE[ai] in finish


@pytest.mark.parametrize("ai", AIS)
def test_stage_file_matches_what_install_writes(tmp_path, ai):
    written = {str(p.relative_to(tmp_path)) for p in installer.install(tmp_path, ai, force=False)}
    for stage in STAGES:
        assert _STEP_FILE[ai].replace("<step>", stage) in written
    assert not any(n in w for w in written for n in ("_finish", "_goto", "_rules"))


@pytest.mark.parametrize("ai", AIS)
def test_reinit_removes_old_stage_prompts_and_keeps_user_files(tmp_path, ai):
    for old in OLD_STAGES:  # what an earlier finetune-kit wrote
        p = tmp_path / _STEP_FILE[ai].replace("<step>", old)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("old prompt")
    agent_dir = tmp_path / installer.AGENTS[ai].commands
    if installer.AGENTS[ai].kind == "skill":
        mine = agent_dir / "finetune-kit-setup" / "my-notes.md"  # user file inside an old folder
    else:
        mine = agent_dir / "finetune-kit.setup.notes.md"
    mine.write_text("mine")
    other = agent_dir / "my-own-command.md"
    other.write_text("mine too")

    installer.install(tmp_path, ai, force=False)

    for old in OLD_STAGES:
        assert not (tmp_path / _STEP_FILE[ai].replace("<step>", old)).exists(), old
        if installer.AGENTS[ai].kind == "skill" and old != "setup":
            assert not (agent_dir / f"finetune-kit-{old}").exists(), old
    assert mine.read_text() == "mine" and other.read_text() == "mine too"
    for stage in STAGES:
        assert (tmp_path / _STEP_FILE[ai].replace("<step>", stage)).is_file()


def test_old_templates_are_gone():
    root = resources.files("finetune_kit") / "templates" / "commands"
    for old in OLD_STAGES:
        assert not (root / f"{old}.md").is_file()


@pytest.mark.parametrize("ai,path", [
    ("claude", ".claude/skills/finetune-kit-choose-data/SKILL.md"),
    ("codex", ".agents/skills/finetune-kit-choose-data/SKILL.md"),
    ("cursor", ".cursor/commands/finetune-kit.choose-data.md"),
])
def test_choose_data_prompt_mentions_data_sources(tmp_path, ai, path):
    installer.install(tmp_path, ai, force=False)
    text = (tmp_path / path).read_text()
    assert "inspect-data" in text and "--source hf:" in text and "--columns" in text


@pytest.mark.parametrize("ai", AIS)
def test_choose_loss_offers_catalog_losses_and_says_when_a_new_round_starts(ai):
    text = installer.render("choose-loss", ai)
    steps = text.split("## Steps", 1)[1].split("## Finish", 1)[0]
    assert "`losses` list" in steps and "★" in steps
    assert "wait" in steps.lower()
    assert "finetune-kit set losses <exp>" in steps
    assert "new round" in text
    assert "Which of these may I change?" not in text


def test_choose_hyperparameters_uses_the_chosen_losses():
    text = installer.render("choose-hyperparameters", "codex")
    assert "finetune-kit space commit <exp> --tunable" in text
    assert "$finetune-kit-choose-loss" in text  # losses come from the step before
    assert "Which losses may research try?" not in text


def test_choose_learning_rate_runs_the_scan_and_waits():
    text = installer.render("choose-learning-rate", "codex")
    assert "finetune-kit set learning-rate <exp> default" in text
    assert "finetune-kit set learning-rate <exp> scan" in text
    assert "finetune-kit lr-scan <exp>" in text and "--timeout 540" in text


def test_choose_experiments_asks_budget_seeds_and_subset():
    text = installer.render("choose-experiments", "codex")
    for opt in ("--max-runs", "--max-hours", "--seeds", "--subset"):
        assert opt in text


def test_start_experiment_and_finish_specifics():
    research = installer.render("start-experiment", "claude")
    assert "Never ask the human whether to continue" in research
    assert "log.md" in research and "finetune-kit compare" in research and "--stage noise" in research
    finish = installer.render("finish", "claude")
    assert "exactly once" in finish and "answer yes" in finish


def test_rules_have_keep_formula_and_stop_rule():
    rules = (resources.files("finetune_kit") / "templates/memory/research-rules.md").read_text()
    assert "noise = σ · sqrt(2/N)" in rules
    assert "Δ > max(2·noise, min_delta)" in rules
    assert "σ_guard · sqrt(2/N)" in rules
    assert "|Δ| ≤ 2·noise" in rules
    assert "10 non-keeps in a row" in rules


def test_rules_name_the_new_steps_not_the_old_stages():
    rules = _template_text("memory", "research-rules.md")
    assert "never skip a step" in rules.lower()
    assert "choose-hyperparameters" in rules
    for old in ("hparams stage", "the check stage", "never skip a stage"):
        assert old not in rules


def test_cursor_command_frontmatter_parses(tmp_path):
    installer.install(tmp_path, "cursor", force=False)
    fm = _frontmatter((tmp_path / ".cursor/commands/finetune-kit.start-experiment.md").read_text())
    assert fm["description"] == installer.DESCRIPTIONS["start-experiment"]


def test_descriptions_cover_every_step_in_plain_words():
    assert set(installer.DESCRIPTIONS) == set(STAGES)
    assert "doesn't know where to start" in installer.DESCRIPTIONS["start"]
    for d in installer.DESCRIPTIONS.values():
        assert ": " not in d  # plain YAML scalar in the frontmatter
        assert "\n" not in d and d.startswith(tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))


@pytest.mark.parametrize("settings", [{"hooks": []}, {"hooks": {"PreToolUse": {}}}, {"hooks": {"PreToolUse": [{"hooks": 5}]}}])
def test_init_rejects_malformed_settings_shapes(tmp_path, settings):
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude/settings.json").write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="settings.json"):
        installer.install(tmp_path, "claude", force=False)


def test_agents_md_and_cursor_rules_list_the_steps(tmp_path):
    installer.install(tmp_path, "codex", force=False)
    installer.install(tmp_path, "cursor", force=False)
    agents = (tmp_path / "AGENTS.md").read_text()
    mdc = (tmp_path / ".cursor/rules/finetune-kit.mdc").read_text()
    assert "$finetune-kit-start" in agents and "/finetune-kit.start" in mdc
    for text in (agents, mdc):
        assert "choose-hyperparameters" in text and "hparams" not in text


def test_research_summary_names_its_round():
    assert "## Summary (round" in installer.render("start-experiment", "codex")


def test_compare_models_reads_the_cli_not_prose():
    text = installer.render("compare-models", "codex")
    assert "the `bakeoff` field of `experiment.yaml`" in text
    assert "bakeoff choose <exp> <model>" in text
    assert "prose" not in text


def test_finish_says_what_to_do_on_no_and_requires_a_yes():
    text = installer.render("check-data", "codex").split("## Finish", 1)[1]
    assert "If the user says no, stop" in text
    assert "Never start a step the user has not agreed to" in text


def test_order_check_handles_a_missing_experiment():
    assert "no argument" in installer.render("check-data", "codex").split("## How to work", 1)[0]


def _template_text(*parts):
    return (resources.files("finetune_kit") / "templates").joinpath(*parts).read_text(encoding="utf-8")


# ---- "Before you start": the shared order check ----

def _before(stage, ai="codex"):
    return installer.render(stage, ai).split("## Before you start", 1)[1].split("## How to work", 1)[0]


@pytest.mark.parametrize("ai", AIS)
def test_choose_hyperparameters_order_check_is_conditional(ai):
    before = _before("choose-hyperparameters", ai)
    assert "This is the `choose-hyperparameters` step." in before
    flat = " ".join(before.split())
    assert ("If `stage` is `choose-hyperparameters` or a later step (`choose-learning-rate`, "
            "`choose-experiments`, `start-experiment`, `report` or `finish`), continue with this step") in flat
    assert ("Otherwise, if `stage` is an earlier step, do not start this step: that step is not "
            "done yet. Earlier steps: `choose-task`, `choose-data`, `choose-model`, `choose-goal`, "
            "`check-data`, `compare-models` or `choose-loss`.") in flat
    assert "finetune-kit next <exp>" in before and "Shall I start" in before
    assert "Step <number> of 14" in before


@pytest.mark.parametrize("stage", GATED)
def test_every_order_check_names_its_step_and_gates_on_no_write_up(stage):
    before = _before(stage)
    assert f"This is the `{stage}` step." in before
    assert "{{" not in before
    if stage != "choose-task":  # choose-task alone first checks that finetune-kit is installed
        assert "must exist" not in before
        assert "Otherwise, if `stage` is an earlier step" in before


def test_finish_goes_on_only_when_next_names_finish():
    before = " ".join(_before("finish").split())
    assert "continue with this step only if `stage` is exactly `finish`" in before
    assert "If `stage` is `finish` or a later step" not in before


def test_report_interim_names_the_round():
    assert "`## Summary (round N)`" in installer.render("report", "codex")


def test_compare_models_after_start_needs_a_new_experiment():
    text = installer.render("compare-models", "codex")
    assert "needs a new experiment" in text and "`bakeoff choose` will refuse" in text


def test_choose_task_with_an_experiment_in_progress_asks_first():
    text = installer.render("choose-task", "codex")
    assert "in progress" in text and "new experiment" in text


# ---- fix round 1 ----

@pytest.mark.parametrize("ai", AIS)
def test_choose_hyperparameters_sends_a_new_round_to_choose_loss(ai):
    text = installer.render("choose-hyperparameters", ai)
    before = " ".join(text.split("## How to work", 1)[0].split())
    assert "A new round always starts with the loss question." in before
    assert "If `next.stage` from the order check is `choose-hyperparameters`, a new round is open" in before
    assert "Do not ask about settings and do not commit anything here" in before
    assert f"Next: Step 7 of 14 — Choose the loss (a new round starts with the loss question). Command: {_PREFIX[ai]}choose-loss" in before
    assert "after choose-loss" in installer.DESCRIPTIONS["choose-hyperparameters"]


def test_descriptions_do_not_overlap_on_starting():
    d = installer.DESCRIPTIONS
    assert "doesn't know where to start" in d["start"] and "overview" in d["start"]
    assert "begin a new experiment" in d["choose-task"]
    assert "once the setup steps are done" in d["start-experiment"]


def test_status_checks_install_and_never_starts_a_second_loop():
    text = installer.render("status", "codex")
    before = text.split("## Before you start", 1)[1].split("## How to work", 1)[0]
    assert "research-rules.md` must exist" in before and "finetune-kit version" in before
    flat = " ".join(text.split())
    assert "Never offer to start a second research loop while one runs" in flat
    assert "If step 5 above found research running, do not offer any step" in flat
    assert "$finetune-kit-start-experiment` resumes it" in flat


@pytest.mark.parametrize("stage", [s for s in STAGES if s != "start"])  # start explains its own list
def test_inner_and_global_step_references_are_distinct(stage):
    import re
    text = installer.render(stage, "codex")
    flat = " ".join(l for l in text.splitlines() if not l.startswith("#")).split()
    flat = " ".join(flat)
    # an inner step is "step N below/above/of the loop"; a global one is "Step N of 14"
    for m in re.finditer(r"\b[Ss]teps? \d+(?:[–-]\d+)?", flat):
        assert flat[m.end():].startswith((" of 14", " below", " above", " of the loop")), \
            (stage, flat[m.start() - 40:m.end() + 20])


def test_init_message_explains_why_rules_are_kept(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    installer.install(tmp_path, "claude", force=False)
    (tmp_path / ".finetune-kit/memory/research-rules.md").write_text("older wording")
    r = runner.invoke(app, ["init", "--ai", "claude", "--here"])
    assert r.exit_code == 0, r.output
    assert "experiments check it for changes" in r.output and "New projects get the updated wording" in r.output
    assert "your edited" not in r.output
    assert (tmp_path / ".finetune-kit/memory/research-rules.md").read_text() == "older wording"


@pytest.mark.parametrize("stage", ["compare-models", "choose-learning-rate"])
def test_first_training_asks_sample_size_and_repeats_early(stage):
    text = installer.render(stage, "codex")
    steps = " ".join(text.split("## Steps", 1)[1].split("## Finish", 1)[0].split())
    assert "n_rows.train" in steps and "5,000 rows" in steps
    assert "smaller sample" in steps and "How many repeats per experiment?" in steps
    assert "finetune-kit set experiments <exp> --subset <rows> --seeds <N>" in steps
    # the early question comes before the training command
    first_training = "finetune-kit bakeoff <exp> --models" if stage == "compare-models" else "finetune-kit lr-scan <exp>"
    assert steps.index("set experiments") < steps.index(first_training)


def test_choose_experiments_shows_fixed_sample_and_repeats():
    text = " ".join(installer.render("choose-experiments", "codex").split())
    assert "already fixed" in text and "lr-scan" in text and "bakeoff" in text
    assert "n_rows.train" in text


def test_choose_loss_sends_a_recommit_to_the_settings_step():
    text = " ".join(installer.render("choose-loss", "codex").split())
    assert '"recommit": true' in text and "Step 8 of 14 — Choose the settings" in text


@pytest.mark.parametrize("stage", STAGES)
def test_prompt_lines_are_wrapped(stage):
    raw = (resources.files("finetune_kit") / "templates" / "commands" / f"{stage}.md").read_text()
    for line in raw.splitlines():
        if "Step " in line and " of 14" in line and not line.lstrip().startswith("`"):
            assert len(line) <= 102, (stage, line)
        assert line.strip() not in ("whether to", "the", "below)."), (stage, line)


def test_init_message_says_how_to_update_the_rules(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    installer.install(tmp_path, "claude", force=False)
    (tmp_path / ".finetune-kit/memory/research-rules.md").write_text("older wording")
    r = runner.invoke(app, ["init", "--ai", "claude", "--here"])
    assert r.exit_code == 0, r.output
    assert "init again with --force" in r.output and "no experiment is mid-research" in r.output
