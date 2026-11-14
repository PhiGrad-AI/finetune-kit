import json
from typing import NoReturn

import typer

import finetune_kit

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Noise-aware fine-tuning research kit.")


@app.callback()
def _main() -> None:
    """Keeps `app` a command group even while it has a single command."""


def emit(obj: dict) -> None:
    typer.echo(json.dumps(obj))


def fail(msg: str) -> NoReturn:
    typer.echo(json.dumps({"error": msg}))
    raise typer.Exit(2)


@app.command()
def version() -> None:
    """Print the installed version as JSON."""
    emit({"version": finetune_kit.__version__})


if __name__ == "__main__":
    app()
