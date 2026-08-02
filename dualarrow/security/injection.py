from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class InjectionFinding:
    rule: str
    excerpt: str
    severity: str


_PATTERNS: dict[str, re.Pattern[str]] = {
    "ignore_instructions": re.compile(
        r"\b(ignore|disregard|forget)\b.{0,50}\b(previous|system|developer)\b.{0,30}\b(instruction|prompt)s?\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "reveal_prompt": re.compile(
        r"\b(reveal|show|print|repeat)\b.{0,40}\b(system prompt|hidden instructions|developer message)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "tool_override": re.compile(
        r"\b(call|execute|run)\b.{0,30}\b(tool|command|shell|terminal)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "role_impersonation": re.compile(
        r"(^|\n)\s*(system|developer|assistant)\s*:\s*",
        re.IGNORECASE,
    ),
}


def scan_prompt_injection(text: str) -> list[InjectionFinding]:
    findings: list[InjectionFinding] = []
    for name, pattern in _PATTERNS.items():
        for match in pattern.finditer(text):
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            findings.append(
                InjectionFinding(
                    rule=name,
                    excerpt=text[start:end].replace("\n", " "),
                    severity="high" if name in {"ignore_instructions", "role_impersonation"} else "medium",
                )
            )
    return findings
