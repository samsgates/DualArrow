from __future__ import annotations

import re

from dualarrow.config import VerificationConfig
from dualarrow.models import (
    AnswerStatus,
    Citation,
    Claim,
    ClaimStatus,
    ConfidenceBreakdown,
    DraftAnswer,
    EvidenceCandidate,
)
from dualarrow.utils import clamp, tokenize

_NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:%|\b)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_HEADING_LINE_RE = re.compile(r"^\s*#{1,6}\s+.*$", re.MULTILINE)
_NEGATIONS = {"not", "no", "never", "without", "cannot", "can't", "prohibited", "forbidden"}
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was",
    "were", "be", "been", "with", "that", "this", "it", "as", "at", "by", "from",
}


class VerificationOutcome:
    def __init__(
        self,
        *,
        status: AnswerStatus,
        answer: str,
        claims: list[Claim],
        citations: list[Citation],
        confidence: ConfidenceBreakdown,
        warnings: list[str],
    ) -> None:
        self.status = status
        self.answer = answer
        self.claims = claims
        self.citations = citations
        self.confidence = confidence
        self.warnings = warnings


class EvidenceVerifier:
    def __init__(self, config: VerificationConfig) -> None:
        self.config = config

    async def verify(
        self,
        query: str,
        draft: DraftAnswer,
        evidence: list[EvidenceCandidate],
    ) -> VerificationOutcome:
        evidence_by_id = {item.evidence_id: item for item in evidence}
        verified_claims: list[Claim] = []
        warnings: list[str] = []
        used_evidence_ids: list[str] = []

        for claim in draft.claims:
            candidate_ids = [item for item in claim.evidence_ids if item in evidence_by_id]
            if not candidate_ids:
                candidate_ids = [item.evidence_id for item in evidence]
            best_support = 0.0
            best_contradiction = 0.0
            supporting_ids: list[str] = []
            notes: list[str] = []
            for evidence_id in candidate_ids:
                item = evidence_by_id[evidence_id]
                support, contradiction, item_notes = self._compare(claim.text, item.content)
                if support > best_support:
                    best_support = support
                if contradiction > best_contradiction:
                    best_contradiction = contradiction
                if support >= self.config.minimum_support_score and contradiction < 0.5:
                    supporting_ids.append(evidence_id)
                notes.extend(item_notes)

            if best_contradiction >= 0.65:
                status = ClaimStatus.CONTRADICTED
            elif best_support >= self.config.minimum_support_score:
                status = ClaimStatus.SUPPORTED
            elif best_support >= self.config.minimum_support_score * 0.65:
                status = ClaimStatus.PARTIALLY_SUPPORTED
            else:
                status = ClaimStatus.NOT_FOUND
            verified = claim.model_copy(
                update={
                    "status": status,
                    "evidence_ids": supporting_ids,
                    "support_score": clamp(best_support),
                    "contradiction_score": clamp(best_contradiction),
                    "notes": sorted(set(notes)),
                }
            )
            verified_claims.append(verified)
            used_evidence_ids.extend(supporting_ids)

        material_claims = [claim for claim in verified_claims if claim.status != ClaimStatus.DERIVED]
        supported = [claim for claim in material_claims if claim.status == ClaimStatus.SUPPORTED]
        partial = [claim for claim in material_claims if claim.status == ClaimStatus.PARTIALLY_SUPPORTED]
        contradicted = [claim for claim in material_claims if claim.status == ClaimStatus.CONTRADICTED]
        claim_coverage = len(supported) / max(1, len(material_claims))
        conflict_score = max(
            [claim.contradiction_score for claim in material_claims] or [0.0]
        )

        unique_evidence = list(dict.fromkeys(used_evidence_ids))
        citations = [self._citation(evidence_by_id[item]) for item in unique_evidence]
        citation_precision = (
            len(unique_evidence) / max(1, sum(len(claim.evidence_ids) for claim in verified_claims))
            if verified_claims else 0.0
        )
        authority = self._average(
            [evidence_by_id[item].authority_score for item in unique_evidence]
        )
        freshness = self._average(
            [evidence_by_id[item].freshness_score for item in unique_evidence]
        )
        retrieval_coverage = clamp(len(evidence) / 5.0)
        overall = clamp(
            0.22 * retrieval_coverage
            + 0.32 * claim_coverage
            + 0.16 * citation_precision
            + 0.16 * authority
            + 0.14 * freshness
            - 0.35 * conflict_score
        )
        confidence = ConfidenceBreakdown(
            overall=overall,
            retrieval_coverage=retrieval_coverage,
            claim_support=claim_coverage,
            citation_precision=clamp(citation_precision),
            source_authority=authority,
            freshness=freshness,
            conflict_score=conflict_score,
        )

        source_count = len({evidence_by_id[item].document_id for item in unique_evidence})
        if self.config.require_multiple_sources and source_count < 2:
            warnings.append("The configured multiple-source requirement was not satisfied.")
            overall = min(overall, self.config.minimum_confidence - 0.01)
            confidence = confidence.model_copy(update={"overall": max(0.0, overall)})

        if contradicted or conflict_score > self.config.maximum_conflict_score:
            status = AnswerStatus.CONFLICTED_ANSWER
        elif not material_claims or not supported:
            status = AnswerStatus.INSUFFICIENT_EVIDENCE
        elif (
            claim_coverage < self.config.minimum_claim_coverage
            or confidence.overall < self.config.minimum_confidence
        ):
            status = AnswerStatus.VERIFIED_PARTIAL_ANSWER if supported else AnswerStatus.INSUFFICIENT_EVIDENCE
        else:
            status = AnswerStatus.VERIFIED_ANSWER

        if partial:
            warnings.append(f"{len(partial)} claim(s) were only partially supported.")
        unsupported_count = sum(
            1 for claim in material_claims if claim.status == ClaimStatus.NOT_FOUND
        )
        if unsupported_count:
            warnings.append(f"{unsupported_count} claim(s) lacked sufficient evidence.")

        answer = self._safe_answer(draft.answer, verified_claims, status)
        return VerificationOutcome(
            status=status,
            answer=answer,
            claims=verified_claims,
            citations=citations,
            confidence=confidence,
            warnings=warnings,
        )

    def _compare(self, claim: str, evidence: str) -> tuple[float, float, list[str]]:
        cleaned = _HEADING_LINE_RE.sub("", evidence)
        spans = [" ".join(item.split()) for item in _SENTENCE_SPLIT_RE.split(cleaned) if item.strip()]
        if not spans:
            spans = [evidence]
        best_support = 0.0
        best_contradiction = 0.0
        best_notes: list[str] = []
        for span in spans:
            support, contradiction, notes = self._compare_span(claim, span)
            if support > best_support or (support == best_support and contradiction < best_contradiction):
                best_support = support
                best_contradiction = contradiction
                best_notes = notes
        return best_support, best_contradiction, best_notes

    def _compare_span(self, claim: str, evidence: str) -> tuple[float, float, list[str]]:
        claim_tokens = [token for token in tokenize(claim) if token not in _STOP]
        evidence_tokens = set(tokenize(evidence))
        if not claim_tokens:
            return 0.0, 0.0, ["empty_claim_tokens"]
        weighted_hits = sum(
            1.5 if any(char.isdigit() for char in token) else 1.0
            for token in claim_tokens
            if token in evidence_tokens
        )
        weighted_total = sum(
            1.5 if any(char.isdigit() for char in token) else 1.0
            for token in claim_tokens
        )
        support = weighted_hits / max(1.0, weighted_total)
        notes: list[str] = []

        claim_numbers = set(_NUMBER_RE.findall(claim))
        evidence_numbers = set(_NUMBER_RE.findall(evidence))
        if self.config.validate_numbers and claim_numbers:
            missing = claim_numbers - evidence_numbers
            if missing:
                support *= 0.45
                notes.append(f"numbers_not_found:{','.join(sorted(missing))}")

        claim_negated = any(token in _NEGATIONS for token in tokenize(claim))
        evidence_negated = any(token in _NEGATIONS for token in tokenize(evidence))
        contradiction = 0.0
        if claim_negated != evidence_negated and support > 0.55:
            contradiction = min(1.0, support + 0.15)
            notes.append("negation_mismatch")
        return clamp(support), clamp(contradiction), notes

    @staticmethod
    def _average(values: list[float]) -> float:
        return clamp(sum(values) / len(values)) if values else 0.0

    @staticmethod
    def _citation(item: EvidenceCandidate) -> Citation:
        quote = item.content.strip().replace("\n", " ")
        if len(quote) > 260:
            quote = quote[:257].rstrip() + "..."
        return Citation(
            evidence_id=item.evidence_id,
            source_uri=item.source_uri,
            title=item.title,
            heading_path=item.heading_path,
            line_start=item.line_start,
            line_end=item.line_end,
            quote=quote,
        )

    @staticmethod
    def _safe_answer(
        original: str,
        claims: list[Claim],
        status: AnswerStatus,
    ) -> str:
        if status == AnswerStatus.INSUFFICIENT_EVIDENCE:
            return "DualArrow could not find enough verifiable evidence to answer this question."
        if status == AnswerStatus.CONFLICTED_ANSWER:
            supported = [claim.text for claim in claims if claim.status == ClaimStatus.SUPPORTED]
            prefix = "The available evidence conflicts."
            return prefix + (" Supported findings: " + " ".join(supported) if supported else "")
        safe_claims = [
            claim.text
            for claim in claims
            if claim.status in {ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED}
        ]
        return " ".join(safe_claims) if safe_claims else original
