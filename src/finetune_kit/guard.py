"""Claude Code PreToolUse hook (`finetune-kit _guard`): block agent edits to files the CLI owns.

Reads the hook JSON on stdin. Exit 2 with a message on stderr blocks the tool call and shows
the message to the agent; exit 0 lets it through.
"""
import json
import os
import sys
from pathlib import Path

from finetune_kit.paths import kit_home

RUN_DIRS = {"splits", "final", "runs"}
RUN_FILES = {"search-space.yaml", "state.json", "results.tsv", "experiment.yaml", "data-report.json"}
PATH_KEYS = ("file_path", "path", "notebook_path")


def _candidates(path: str, cwd: str) -> list[Path]:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path(cwd or os.getcwd()) / p
    normal = Path(os.path.normpath(p))
    return list(dict.fromkeys([normal, normal.resolve()]))


def _within(p: Path, root: Path) -> bool:
    return p == root or root in p.parents


def _reason(p: Path) -> str | None:
    for home in {kit_home(), Path.home() / ".finetune-kit"}:
        for root in {Path(os.path.normpath(home)), home.resolve()}:
            if _within(p, root):
                return "it is under ~/.finetune-kit/ (the test vault and run queue); only the CLI reads it"
    parts = p.parts
    for i, part in enumerate(parts[:-1]):
        if part == ".finetune-kit" and parts[i + 1] == "memory":
            return "the research rules are locked during research; the user edits them, not the agent"
        if part == "experiments" and len(parts) >= i + 3:
            name = parts[i + 2]
            if name in RUN_DIRS:
                return f"experiments/*/{name}/ is written only by finetune-kit"
            if len(parts) == i + 3 and name in RUN_FILES:
                return f"experiments/*/{name} is written only by finetune-kit"
    return None


def blocked(path: str, cwd: str) -> str | None:
    """Why `path` (relative to `cwd`) is protected, or None when the agent may edit it."""
    for p in _candidates(path, cwd):
        reason = _reason(p)
        if reason:
            return reason
    return None


def main(stdin=None) -> int:
    try:
        event = json.load(stdin or sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 0
    tool_input = event.get("tool_input") if isinstance(event, dict) else None
    if not isinstance(tool_input, dict):
        return 0
    for key in PATH_KEYS:
        path = tool_input.get(key)
        if isinstance(path, str) and path:
            reason = blocked(path, event.get("cwd") or "")
            if reason:
                print(
                    f"finetune-kit: {path} is protected: {reason}. Use `finetune-kit` commands instead "
                    "(see .finetune-kit/memory/research-rules.md).",
                    file=sys.stderr,
                )
                return 2
    return 0
