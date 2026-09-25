"""Resonate CLI entrypoint."""

from __future__ import annotations

import os

# Configure runtime environment before importing any C/C++ acceleration libraries
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


import typer
from rich.console import Console

from resonate.cli.analyze import analyze_cmd
from resonate.cli.check import check_cmd
from resonate.cli.clean import clean_cmd
from resonate.cli.setup_cmd import wizard_cmd
from resonate.cli.status import status_cmd
from resonate.cli.tune_cmd import tune_app

console = Console()


app = typer.Typer(
    name="resonate",
    help="Music taxonomy, acoustic mood tagging, and audio cleaner for Plex.",
    add_completion=False,
)

app.command(name="analyze")(analyze_cmd)
app.command(name="check")(check_cmd)
app.command(name="clean")(clean_cmd)
app.command(name="setup")(wizard_cmd)
app.command(name="status")(status_cmd)
app.add_typer(tune_app, name="tune")

__all__ = ["app"]


if __name__ == "__main__":
    app(prog_name="./resonate")
