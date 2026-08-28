"""CLI command to run interactive setup wizard."""

from typing import Annotated

import typer

from resonate.wizard import run_wizard


def wizard_cmd(
    config: Annotated[
        str,
        typer.Option("--config", "-c", help="Path to config file to save wizard settings"),
    ] = "config.yaml",
) -> None:
    """Run interactive setup wizard to configure Resonate."""
    run_wizard(config_path=config)
