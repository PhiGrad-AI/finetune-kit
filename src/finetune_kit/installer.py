"""`finetune-kit init`: write the stage prompts, rules and templates for one coding agent.

Idempotent: re-running overwrites only finetune-kit's own files (and the marked AGENTS.md
section); user files and experiments/ are never touched. A research-rules.md the user has
edited is kept unless `force`.
"""
import json
import textwrap
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

STAGES = ["setup", "check", "bakeoff", "hparams", "baseline", "research", "analyze", "promote"]
DOCS = ["experiment", "search-space", "baseline", "report"]
RULES = ".finetune-kit/memory/research-rules.md"
START, END = "<!-- finetune-kit:start -->", "<!-- finetune-kit:end -->"
GUARD_COMMAND = "finetune-kit _guard"
GUARD_MATCHER = "Edit|Write|MultiEdit|NotebookEdit"

DESCRIPTIONS = {
    "setup": "Set up a new fine-tuning experiment by choosing the task, dataset, model, metric, seeds and budget. Use when the user wants to start fine-tuning or training a classifier or NER model.",
    "check": "Check the experiment's data for problems and freeze the train/val/test splits. Use after setup, or when the user asks to check or validate their training data.",
    "bakeoff": "Compare 2-3 candidate models on the same data and pick one. Use after the data check when setup chose a model bake-off.",
    "hparams": "Choose which hyperparameters and losses the research may tune. Use after the data check or bake-off, or to open more settings after research plateaus.",
    "baseline": "Run the learning-rate scan and measure the noise floor of the default recipe. Use after choosing hyperparameters, before research.",
    "research": "Run or resume the autonomous fine-tuning research loop until the stop rule. Use after the baseline, or to continue research that was interrupted.",
    "analyze": "Analyze the research results, what helped and what hurt, with an error analysis of the best run. Use when research has stopped or the user asks how the experiments went.",
    "promote": "Retrain the chosen recipe on full data, score the test set once and export the final model. Use when the user is ready to finalize or ship the model.",
}


@dataclass(frozen=True)
class Agent:
    kind: str           # "skill" (folder per stage with SKILL.md) or "command" (one file per stage)
    commands: str       # directory for the stage files, relative to the project root
    invoke: str         # prefix that invokes a stage, replaces {{CMD}}
    label: str


AGENTS = {
    "claude": Agent("skill", ".claude/skills", "/finetune-kit-", "Claude Code"),
    "codex": Agent("skill", ".agents/skills", "$finetune-kit-", "Codex"),
    "cursor": Agent("command", ".cursor/commands", "/finetune-kit.", "Cursor"),
}

POINTER = f"""This project uses finetune-kit for fine-tuning research. Before any fine-tuning work, read
`{RULES}` and follow it. Drive experiments only through `finetune-kit` CLI commands (each
prints one JSON object); never edit files under `.finetune-kit/memory/`, `experiments/*/splits/`,
`experiments/*/runs/`, `experiments/*/final/`, or `experiments/*/{{search-space.yaml,state.json,results.tsv,experiment.yaml,data-report.json}}`,
and never open anything under `~/.finetune-kit/`. The stages, in order: {", ".join(STAGES)}."""


def _template(*parts: str) -> str:
    return (resources.files("finetune_kit") / "templates").joinpath(*parts).read_text(encoding="utf-8")


def _write(path: Path, text: str, written: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    written.append(path)


def stage_file(stage: str, ai: str) -> str:
    """Where `ai` has the prompt of `stage`, relative to the project root. `stage` may be "<stage>"."""
    agent = AGENTS[ai]
    if agent.kind == "skill":
        return f"{agent.commands}/finetune-kit-{stage}/SKILL.md"
    return f"{agent.commands}/finetune-kit.{stage}.md"


def _words(names: list[str]) -> str:
    quoted = [f"`{n}`" for n in names]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " or " + quoted[-1]


def _order_words(stage: str) -> dict[str, str]:
    """What the shared order check (_goto.md) says for `stage`: which stages are earlier, and when to go on."""
    i = STAGES.index(stage)
    earlier, later = STAGES[:i], STAGES[i + 1:]
    if later:
        go_on = (f"If `stage` is `{stage}` or a later stage ({_words(later)}), continue with this stage "
                 "below; its own steps check what is already done.")
    else:
        go_on = (f"You may continue with this stage only if `stage` is exactly `{stage}`. Scoring the test "
                 "set cannot be undone, so never promote when `next` names any other stage. One exception: "
                 "`setup` with the reason \"this experiment is complete; start a new one\" means it was "
                 "promoted already; then do only step 1 under Steps (Already done?): summarize, never promote again.")
    return {
        "{{STAGE_ORDER}}": " → ".join(STAGES),
        "{{STAGE}}": stage,
        "{{EARLIER}}": _words(earlier) if earlier else "none",
        "{{CONTINUE_WHEN}}": textwrap.fill(go_on, width=93, subsequent_indent="   "),
    }


def render(stage: str, ai: str) -> str:
    """The stage prompt for one agent: frontmatter for skills, shared blocks and {{CMD}} filled in."""
    agent = AGENTS[ai]
    body = _template("commands", f"{stage}.md")
    for key, shared in (("{{FINISH}}", "_finish.md"), ("{{ORDER_CHECK}}", "_goto.md")):
        body = body.replace(key, _template("commands", shared).rstrip("\n"))
    for key, value in _order_words(stage).items():
        body = body.replace(key, value)
    body = body.replace("{{CMD}}", agent.invoke).replace("{{STAGE_FILE}}", stage_file("<stage>", ai))
    if agent.kind == "skill":
        head = f"---\nname: finetune-kit-{stage}\ndescription: {DESCRIPTIONS[stage]}\n---\n\n"
    else:
        head = f"---\ndescription: {DESCRIPTIONS[stage]}\n---\n\n"
    return head + body


def _merge_settings(path: Path, written: list[Path]) -> None:
    settings: dict = {}
    if path.is_file():
        try:
            settings = json.loads(path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} is not valid JSON ({e}); fix it and run init again") from None
        if not isinstance(settings, dict):
            raise ValueError(f"{path} must hold a JSON object; fix it and run init again")
    hooks = settings.setdefault("hooks", {})
    pre = hooks.setdefault("PreToolUse", []) if isinstance(hooks, dict) else None
    if not isinstance(pre, list) or any(
        isinstance(e, dict) and not isinstance(e.get("hooks", []), list) for e in pre
    ):
        raise ValueError(
            f"{path}: \"hooks\" must be an object whose \"PreToolUse\" is a list of "
            "{\"matcher\", \"hooks\": [...]} entries; fix it and run init again"
        )
    present = any(
        h.get("command") == GUARD_COMMAND
        for entry in pre if isinstance(entry, dict)
        for h in entry.get("hooks", []) if isinstance(h, dict)
    )
    if present:
        return
    pre.append({"matcher": GUARD_MATCHER, "hooks": [{"type": "command", "command": GUARD_COMMAND}]})
    _write(path, json.dumps(settings, indent=2) + "\n", written)


def _agents_md(path: Path, written: list[Path]) -> None:
    section = f"{START}\n## finetune-kit\n\n{POINTER}\n\nInvoke a stage with `$finetune-kit-<stage>`, starting with `$finetune-kit-setup`.\n{END}"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    if START in text and END in text and text.index(START) < text.index(END):
        before, rest = text.split(START, 1)
        after = rest.split(END, 1)[1]
        new = before + section + after
    else:
        new = (text.rstrip("\n") + "\n\n" if text.strip() else "") + section + "\n"
    if new != text:
        _write(path, new, written)


def install(root: Path, ai: str, force: bool) -> list[Path]:
    """Install finetune-kit for `ai` into `root`; returns the files written."""
    if ai not in AGENTS:
        raise ValueError(f"--ai must be one of {', '.join(AGENTS)}, not {ai!r}")
    root = Path(root)
    agent = AGENTS[ai]
    written: list[Path] = []
    if ai == "claude":  # first: an unreadable settings.json stops init before anything is written
        _merge_settings(root / ".claude" / "settings.json", written)

    rules = root / RULES
    shipped = _template("memory", "research-rules.md")
    if not rules.is_file() or (force and rules.read_text(encoding="utf-8") != shipped):
        _write(rules, shipped, written)
    for doc in DOCS:
        _write(root / ".finetune-kit/templates" / f"{doc}.md", _template("docs", f"{doc}.md"), written)
    (root / "experiments").mkdir(parents=True, exist_ok=True)
    keep = root / "experiments" / ".gitkeep"
    if not keep.exists():
        _write(keep, "", written)

    for stage in STAGES:
        _write(root / stage_file(stage, ai), render(stage, ai), written)

    if ai == "codex":
        _agents_md(root / "AGENTS.md", written)
    elif ai == "cursor":
        mdc = f"---\ndescription: finetune-kit research rules\nglobs:\nalwaysApply: true\n---\n\n{POINTER}\n\nRun a stage with `/finetune-kit.<stage>`, starting with `/finetune-kit.setup`.\n"
        _write(root / ".cursor/rules/finetune-kit.mdc", mdc, written)
    return written


def rules_kept(root: Path) -> bool:
    """True when the project's research-rules.md differs from the shipped one (kept without --force)."""
    rules = Path(root) / RULES
    return rules.is_file() and rules.read_text(encoding="utf-8") != _template("memory", "research-rules.md")
