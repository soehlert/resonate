"""Beets CLI integration module to modify audio file metadata tags."""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


class BeetsTagger:
    """Update file mood metadata using Beets CLI."""

    def __init__(self, binary_path: str = "beet", enabled: bool = True) -> None:
        """Initialize BeetsTagger with binary path and enabled flag."""
        self.binary_path = binary_path
        self.enabled = enabled

    def update_file_mood(self, file_path: str, mood: str, dry_run: bool = False) -> bool:
        """Update mood tag for specified file via beets command line interface."""
        if not self.enabled:
            logger.info("Beets tagging is disabled.")
            return False

        if not os.path.exists(file_path):
            logger.warning(f"File not found for Beets tagging: {file_path}")
            return False

        cmd = [
            self.binary_path,
            "modify",
            "-y",
            f"path:{file_path}",
            f"mood={mood}",
        ]

        if dry_run:
            logger.info(f"[DRY RUN] Would execute command: {' '.join(cmd)}")
            return True

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                logger.info(f"Successfully updated mood tags on {file_path}")
                return True
            else:
                logger.warning(
                    f"Beets command failed with exit code {result.returncode}: {result.stderr}"
                )
                return False
        except FileNotFoundError:
            logger.warning(f"Beets binary not found at '{self.binary_path}'")
            return False
        except Exception as err:
            logger.warning(f"Failed to execute Beets command: {err}")
            return False
