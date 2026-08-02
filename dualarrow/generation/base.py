from __future__ import annotations

from typing import Protocol

from dualarrow.models import DraftAnswer, EvidenceCandidate


class AnswerGenerator(Protocol):
    async def generate(self, query: str, evidence: list[EvidenceCandidate]) -> DraftAnswer:
        ...
