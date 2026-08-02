from __future__ import annotations

import contextvars

_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "dualarrow_trace_id", default=None
)


def set_trace_id(value: str) -> None:
    _trace_id.set(value)


def get_trace_id() -> str | None:
    return _trace_id.get()
