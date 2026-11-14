import os
from pathlib import Path


def kit_home() -> Path:
    env = os.environ.get("FINETUNE_KIT_HOME")
    return Path(env) if env else Path.home() / ".finetune-kit"


def vault_dir(uid: str) -> Path:
    return kit_home() / "vault" / uid


def find_experiment(name_or_path: str, root: Path | None = None) -> Path:
    """Resolve `001`, `001-demo-exp`, a bare name (`demo-exp`) or a path to an experiment dir."""
    root = Path.cwd() if root is None else Path(root)
    p = Path(name_or_path)
    if (p.is_absolute() or len(p.parts) > 1) and p.is_dir():
        return p
    exp_root = root / "experiments"
    dirs = sorted(d for d in exp_root.iterdir() if d.is_dir()) if exp_root.is_dir() else []
    exact = [d for d in dirs if d.name == name_or_path]
    if exact:
        return exact[0]
    matches = [
        d for d in dirs
        if d.name.startswith(f"{name_or_path}-") or d.name.partition("-")[2] == name_or_path
    ]
    if not matches:
        raise FileNotFoundError(f"no experiment matching {name_or_path!r} under {exp_root}")
    if len(matches) > 1:
        raise ValueError(
            f"{name_or_path!r} is ambiguous: " + ", ".join(d.name for d in matches)
        )
    return matches[0]
