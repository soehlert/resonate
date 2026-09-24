"""Decision tracing utilities for engine pipeline and rule execution."""

from __future__ import annotations


class DecisionTracer:
    """Collects and formats diagnostic decision events across engine phases."""

    def __init__(self, messages: list[str] | None = None, enabled: bool = True) -> None:
        """Initialize tracer wrapping an optional message list."""
        self.messages: list[str] = messages if messages is not None else []
        self.enabled = enabled

    def record(self, message: str) -> None:
        """Append an arbitrary diagnostic message."""
        if self.enabled:
            self.messages.append(message)

    def accept(self, source: str, item: str, score: float | None = None) -> None:
        """Record an accepted classification candidate."""
        if self.enabled:
            details = f" (score={score:.2f})" if score is not None else ""
            self.messages.append(f"{source} accepted: '{item}'{details}")

    def skip(self, source: str, item: str, reason: str) -> None:
        """Record a skipped candidate due to threshold or gate."""
        if self.enabled:
            self.messages.append(f"{source} '{item}' skipped: {reason}")

    def reject(self, item: str, reason: str) -> None:
        """Record a candidate rejected due to context or conflict."""
        if self.enabled:
            self.messages.append(f"Rejected '{item}': {reason}")

    def drop(self, item: str, rule: str) -> None:
        """Record an item dropped by exclusion or conflict rule."""
        if self.enabled:
            self.messages.append(f"Dropped '{item}': {rule}")
