from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

_WORD_RE = re.compile(r"[\w][\w\-./:+]*", re.UNICODE)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(*parts: str, prefix: str = "") -> str:
    raw = "\x1f".join(parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}{digest}"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def utcnow() -> datetime:
    return datetime.now(UTC)


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in _WORD_RE.findall(text)]


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
