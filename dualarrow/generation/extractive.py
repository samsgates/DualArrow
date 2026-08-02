from __future__ import annotations

import re

from dualarrow.models import Claim, DraftAnswer, EvidenceCandidate
from dualarrow.utils import stable_id, tokenize

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_HEADING_LINE_RE = re.compile(r"^\s*#{1,6}\s+.*$", re.MULTILINE)
_IDENTIFIER_RE = re.compile(r"\b[A-Z]{1,8}-\d{2,10}[A-Z0-9-]*\b")
_STOPWORDS = {
    "a", "an", "the", "what", "which", "who", "when", "where", "why", "how",
    "is", "are", "was", "were", "be", "been", "should", "would", "could", "at",
    "after", "before", "for", "of", "to", "in", "on", "and", "or", "it", "this",
}


def _matches(query_token: str, candidate_tokens: set[str]) -> bool:
    if query_token in candidate_tokens:
        return True
    if len(query_token) >= 4:
        stem = query_token[:4]
        return any(token.startswith(stem) or stem in token for token in candidate_tokens)
    return False


class ExtractiveGenerator:
    """Safe offline generator that only quotes retrieved evidence sentences."""

    async def generate(self, query: str, evidence: list[EvidenceCandidate]) -> DraftAnswer:
        all_query_tokens = set(tokenize(query))
        query_identifiers = {item.casefold() for item in _IDENTIFIER_RE.findall(query)}
        query_tokens = {
            token for token in all_query_tokens
            if token not in _STOPWORDS and token not in query_identifiers
        }
        selected: list[tuple[float, str, str]] = []
        used: set[str] = set()
        for item in evidence:
            candidate_context = " ".join([
                item.title,
                " ".join(item.heading_path),
                item.content,
                str(item.metadata.get("product_models", "")),
            ]).casefold()
            if query_identifiers and not any(identifier in candidate_context for identifier in query_identifiers):
                continue
            cleaned = _HEADING_LINE_RE.sub("", item.content)
            cleaned = re.sub(r"^\s*[-*]\s+\[[^]]+\]\([^)]+\)\s*$", "", cleaned, flags=re.MULTILINE)
            sentences = [" ".join(s.split()) for s in _SENTENCE_RE.split(cleaned) if s.strip()]
            heading_tokens = set(tokenize(" ".join(item.heading_path)))
            for sentence in sentences:
                sentence_tokens = set(tokenize(sentence))
                contextual_tokens = sentence_tokens | heading_tokens
                matched = sum(1 for token in query_tokens if _matches(token, contextual_tokens))
                if query_tokens and matched == 0:
                    continue
                overlap = matched / max(1, len(query_tokens))
                score = 0.82 * overlap + 0.18 * item.normalized_score
                normalized = sentence.casefold()
                if score > 0.12 and normalized not in used:
                    used.add(normalized)
                    selected.append((score, sentence, item.evidence_id))
        selected.sort(key=lambda item: item[0], reverse=True)
        selected = selected[:2]
        if not selected:
            return DraftAnswer(answer="", claims=[])
        claims = [
            Claim(
                claim_id=stable_id(sentence, evidence_id, prefix="claim_"),
                text=sentence,
                evidence_ids=[evidence_id],
            )
            for _, sentence, evidence_id in selected
        ]
        answer = " ".join(
            f"{sentence} [{index}]" for index, (_, sentence, _) in enumerate(selected, start=1)
        )
        return DraftAnswer(answer=answer, claims=claims)
