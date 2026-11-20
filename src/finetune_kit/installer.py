"""`finetune-kit init`: write the step prompts, rules and templates for one coding agent.

Idempotent: re-running overwrites only finetune-kit's own files (and the marked AGENTS.md
section); user files and experiments/ are never touched. A research-rules.md the user has
edited is kept unless `force`.
"""
import json
import textwrap
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

STAGES = ["start", "choose-task", "choose-data", "choose-model", "choose-goal", "check-data", "compare-models",
          "choose-loss", "choose-hyperparameters", "choose-learning-rate", "choose-experiments",
          "start-experiment", "status", "report", "finish"]
# What `finetune-kit next` can name, in order (start is the introduction, status runs any time).
ORDER = [s for s in STAGES if s not in ("start", "status")]
# Stage prompts written by earlier versions; `init` removes exactly these files.
OLD_STAGES = ["setup", "check", "bakeoff", "hparams", "baseline", "research", "analyze", "promote"]
DOCS = ["experiment", "search-space", "baseline", "report"]
RULES = ".finetune-kit/memory/research-rules.md"
START, END = "<!-- finetune-kit:start -->", "<!-- finetune-kit:end -->"
GUARD_COMMAND = "finetune-kit _guard"
GUARD_MATCHER = "Edit|Write|MultiEdit|NotebookEdit"

DESCRIPTIONS = {
    "start": "Start here. Shows every fine-tuning step in plain words and where you are. Use when the user wants to fine-tune or train a model but doesn't know where to start, or wants the overview of the steps.",
    "choose-task": "Step 1 of 14. Choose what the model should learn (label whole texts, or tag words in them) and create the experiment. Use when the user wants to begin a new experiment.",
    "choose-data": "Step 2 of 14. Choose the training data (one file, three files or a Hugging Face dataset) and which column is which. Use when the user wants to pick or change the dataset for the experiment.",
    "choose-model": "Step 3 of 14. Choose the model to fine-tune, or 2-3 models to compare. Use when the user asks which model to use or wants to change the model.",
    "choose-goal": "Step 4 of 14. Choose the score to improve and an optional score that must not get worse. Use when the user talks about the metric, what a good result is, or which mistakes matter most.",
    "check-data": "Step 5 of 14. Check the data for problems and split it into training, validation and a locked-away test part. Use when the user asks to check, validate or split their training data.",
    "compare-models": "Step 6 of 14. Train the 2-3 chosen models side by side and keep the best one. Use when the user asked to compare models or wants a model bake-off.",
    "choose-loss": "Step 7 of 14. Choose the losses (ways of measuring mistakes) to experiment with; research tries each and keeps the best, or start another round with other losses. Use when the user asks about losses or wants to try more after a report.",
    "choose-hyperparameters": "Step 8 of 14, after choose-loss. Choose which training settings (hyperparameters) research may change. Use when the user asks what can be tuned; to open more settings in a new round, start with choose-loss.",
    "choose-learning-rate": "Step 9 of 14. Use the model's usual learning rate or run a short test to find a better one. Use when the user asks about the learning rate or learning speed.",
    "choose-experiments": "Step 10 of 14. Choose how many experiments and hours research may use, seeds per experiment and an optional smaller subset. Use when the user talks about budget, time or how many runs.",
    "start-experiment": "Step 11 of 14. Measure the starting point, then run or resume the experiments on their own until they stop. Use once the setup steps are done and the user wants to start, continue or resume the experiments.",
    "status": "Step 12 of 14. Show how the experiments are going (how many ran, the best score so far, the budget left). Use when the user asks for progress, status or how it is going.",
    "report": "Step 13 of 14. Explain what was tried, what helped and what didn't, and the mistakes the best model makes. Use when the experiments have stopped or the user asks for results or a report.",
    "finish": "Step 14 of 14. Train the final model on all the data, score the test set once and export it with predict.py. Use when the user is ready to finish, finalize or ship the model.",
}


@dataclass(frozen=True)
class Agent:
    kind: str           # "skill" (folder per step with SKILL.md) or "command" (one file per step)
    commands: str       # directory for the step files, relative to the project root
    invoke: str         # prefix that invokes a step, replaces {{CMD}}
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
and never open anything under `~/.finetune-kit/`. The steps, in order: {", ".join(STAGES)}."""


def _template(*parts: str) -> str:
    return (resources.files("finetune_kit") / "templates").joinpath(*parts).read_text(encoding="utf-8")


def _write(path: Path, text: str, written: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    written.append(path)


def stage_file(stage: str, ai: str) -> str:
    """Where `ai` has the prompt of `stage`, relative to the project root. `stage` may be "<step>"."""
    agent = AGENTS[ai]
    if agent.kind == "skill":
        return f"{agent.commands}/finetune-kit-{stage}/SKILL.md"
    return f"{agent.commands}/finetune-kit.{stage}.md"


def _words(names: list[str]) -> str:
    quoted = [f"`{n}`" for n in names]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " or " + quoted[-1]


def _order_words(stage: str) -> dict[str, str]:
    """What the shared order check (_goto.md) says for `stage`: which steps are earlier, and when to go on."""
    i = ORDER.index(stage)
    earlier, later = ORDER[:i], ORDER[i + 1:]
    if stage == "finish":
        go_on = ("You may continue with this step only if `stage` is exactly `finish`. Scoring the test "
                 "set cannot be undone, so never finish when `next` names any other step. One exception: "
                 "`choose-task` with the reason \"this experiment is complete; start a new one\" means it "
                 "was finished already; then do only step 1 below (Already done?): summarize, never "
                 "score the test set again.")
    else:
        go_on = (f"If `stage` is `{stage}` or a later step ({_words(later)}), continue with this step "
                 "below; its own steps check what is already done.")
    def fill(text: str, indent: str = "") -> str:
        return textwrap.fill(text, width=96, initial_indent=indent, subsequent_indent=indent or "   ",
                             break_on_hyphens=False, break_long_words=False).lstrip()

    return {
        "{{STAGE_ORDER}}": textwrap.fill(" → ".join(ORDER), width=96, break_on_hyphens=False),
        "{{STAGE}}": stage,
        "{{EARLIER}}": fill(_words(earlier) if earlier else "none", "   "),
        "{{CONTINUE_WHEN}}": fill(go_on),
    }


def render(stage: str, ai: str) -> str:
    """The step prompt for one agent: frontmatter for skills, shared blocks and {{CMD}} filled in."""
    agent = AGENTS[ai]
    body = _template("commands", f"{stage}.md")
    for key, shared in (("{{FINISH}}", "_finish.md"), ("{{ORDER_CHECK}}", "_goto.md"), ("{{HOW_TO_WORK}}", "_rules.md")):
        body = body.replace(key, _template("commands", shared).rstrip("\n"))
    if stage in ORDER:
        for key, value in _order_words(stage).items():
            body = body.replace(key, value)
    body = body.replace("{{CMD}}", agent.invoke).replace("{{STEP_FILE}}", stage_file("<step>", ai))
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
    section = f"{START}\n## finetune-kit\n\n{POINTER}\n\nInvoke a step with `$finetune-kit-<step>`, starting with `$finetune-kit-start`.\n{END}"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    if START in text and END in text and text.index(START) < text.index(END):
        before, rest = text.split(START, 1)
        after = rest.split(END, 1)[1]
        new = before + section + after
    else:
        new = (text.rstrip("\n") + "\n\n" if text.strip() else "") + section + "\n"
    if new != text:
        _write(path, new, written)


def _remove_old_prompts(root: Path, ai: str) -> None:
    """Delete the stage prompts earlier versions wrote (only those files); keep anything else.
    A skill folder is removed only when nothing of the user's is left in it."""
    for old in OLD_STAGES:
        path = root / stage_file(old, ai)
        if path.is_file():
            path.unlink()
        if AGENTS[ai].kind == "skill" and path.parent.is_dir() and not any(path.parent.iterdir()):
            path.parent.rmdir()


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

    _remove_old_prompts(root, ai)
    for stage in STAGES:
        _write(root / stage_file(stage, ai), render(stage, ai), written)

    if ai == "codex":
        _agents_md(root / "AGENTS.md", written)
    elif ai == "cursor":
        mdc = f"---\ndescription: finetune-kit research rules\nglobs:\nalwaysApply: true\n---\n\n{POINTER}\n\nRun a step with `/finetune-kit.<step>`, starting with `/finetune-kit.start`.\n"
        _write(root / ".cursor/rules/finetune-kit.mdc", mdc, written)
    return written


def rules_kept(root: Path) -> bool:
    """True when the project's research-rules.md differs from the shipped one (kept without --force)."""
    rules = Path(root) / RULES
    return rules.is_file() and rules.read_text(encoding="utf-8") != _template("memory", "research-rules.md")
