"""Decision tracing utilities for engine pipeline and rule execution."""

from __future__ import annotations

from resonate.models import TraceAction, TraceEvent


class DecisionTracer:
    """Collects and formats diagnostic decision events across engine phases."""

    def __init__(
        self,
        events: list[TraceEvent] | None = None,
        messages: list[str] | None = None,
        enabled: bool = True,
    ) -> None:
        """Initialize tracer wrapping optional event or message lists."""
        self.events: list[TraceEvent] = events if events is not None else []
        self._legacy_messages = messages
        self.enabled = enabled

    @property
    def messages(self) -> list[str]:
        """Return list of formatted string messages for backwards compatibility."""
        return [e.message for e in self.events]

    def _append(self, event: TraceEvent) -> None:
        if self.enabled:
            self.events.append(event)
            if self._legacy_messages is not None:
                self._legacy_messages.append(event.message)

    def record(self, message: str, action: TraceAction = TraceAction.INFO) -> None:
        """Append an arbitrary diagnostic message."""
        self._append(TraceEvent(action=action, message=message))

    def accept(self, source: str, item: str, score: float | None = None) -> None:
        """Record an accepted classification candidate."""
        details = f" (score={score:.2f})" if score is not None else ""
        self._append(
            TraceEvent(
                action=TraceAction.ACCEPT,
                message=f"{source} accepted: '{item}'{details}",
            )
        )

    def skip(self, source: str, item: str, reason: str) -> None:
        """Record a skipped candidate due to threshold or gate."""
        self._append(
            TraceEvent(
                action=TraceAction.SKIP,
                message=f"{source} '{item}' skipped: {reason}",
            )
        )

    def reject(self, item: str, reason: str) -> None:
        """Record a candidate rejected due to context or conflict."""
        self._append(
            TraceEvent(
                action=TraceAction.REJECT,
                message=f"Rejected '{item}': {reason}",
            )
        )

    def drop(self, item: str, rule: str) -> None:
        """Record an item dropped by exclusion or conflict rule."""
        self._append(
            TraceEvent(
                action=TraceAction.DROP,
                message=f"Dropped '{item}': {rule}",
            )
        )
