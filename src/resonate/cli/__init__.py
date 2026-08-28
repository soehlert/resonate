"""CLI subcommands package for Resonate."""

from resonate.cli.analyze import analyze_cmd
from resonate.cli.check import check_cmd
from resonate.cli.clean import clean_cmd
from resonate.cli.setup_cmd import wizard_cmd
from resonate.cli.status import status_cmd

__all__ = [
    "analyze_cmd",
    "check_cmd",
    "clean_cmd",
    "status_cmd",
    "wizard_cmd",
]
