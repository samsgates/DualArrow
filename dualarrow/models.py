from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AnswerStatus(StrEnum):
    VERIFIED_ANSWER = "VERIFIED_ANSWER"
    VERIFIED_PARTIAL_ANSWER = "VERIFIED_PARTIAL_ANSWER"
    CONFLICTED_ANSWER = "CONFLICTED_ANSWER"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNAUTHORIZED = "UNAUTHORIZED"
    UNSAFE_SOURCE = "UNSAFE_SOURCE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class ClaimStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS = "AMBIGUOUS"
    DERIVED = "DERIVED"


class SourceDocument(StrictModel):
    document_id: str
    title: str
    content: str
    source_uri: str
    checksum: str
    version: str = "1"
    authority: float = Field(default=0.75, ge=0, le=1)
    status: str = "approved"
    effective_from: datetime | None = None
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    modified_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class KnowledgeUnit(StrictModel):
    unit_id: str
    document_id: str
    view_id: str
    content: str
    source_uri: str
    checksum: str
    title: str = ""
    parent_unit_id: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    line_start: int | None = None
    line_end: int | None = None
    authority: float = Field(default=0.75, ge=0, le=1)
    effective_from: datetime | None = None
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceCandidate(StrictModel):
    evidence_id: str
    unit_id: str
    document_id: str
    content: str
    source_uri: str
    title: str = ""
    heading_path: list[str] = Field(default_factory=list)
    line_start: int | None = None
    line_end: int | None = None
    retrieval_methods: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    normalized_score: float = 0.0
    authority_score: float = 0.75
    freshness_score: float = 1.0
    provenance_score: float = 1.0
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(StrictModel):
    evidence_id: str
    source_uri: str
    title: str = ""
    heading_path: list[str] = Field(default_factory=list)
    line_start: int | None = None
    line_end: int | None = None
    quote: str | None = None


class Claim(StrictModel):
    claim_id: str
    text: str
    status: ClaimStatus = ClaimStatus.NOT_FOUND
    evidence_ids: list[str] = Field(default_factory=list)
    support_score: float = Field(default=0.0, ge=0, le=1)
    contradiction_score: float = Field(default=0.0, ge=0, le=1)
    notes: list[str] = Field(default_factory=list)


class ConfidenceBreakdown(StrictModel):
    overall: float = Field(ge=0, le=1)
    retrieval_coverage: float = Field(ge=0, le=1)
    claim_support: float = Field(ge=0, le=1)
    citation_precision: float = Field(ge=0, le=1)
    source_authority: float = Field(ge=0, le=1)
    freshness: float = Field(ge=0, le=1)
    conflict_score: float = Field(ge=0, le=1)


class QueryRequest(StrictModel):
    query: str = Field(min_length=1, max_length=20_000)
    retrieval_profile: str = "auto"
    knowledge_version: str = "latest"
    include_evidence: bool = True
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    tenant_id: str = "default"
    user_roles: list[str] = Field(default_factory=list)


class DraftAnswer(StrictModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)


class AnswerResult(StrictModel):
    status: AnswerStatus
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[EvidenceCandidate] = Field(default_factory=list)
    confidence: ConfidenceBreakdown
    trace_id: str
    retrieval_profile: str
    warnings: list[str] = Field(default_factory=list)


class IngestionReport(StrictModel):
    run_id: str
    documents_seen: int = 0
    documents_indexed: int = 0
    units_created: int = 0
    documents_deleted: int = 0
    quarantined: int = 0
    errors: list[str] = Field(default_factory=list)
    knowledge_version: str


class RetrievalTrace(StrictModel):
    trace_id: str
    query: str
    normalized_query: str
    selected_profile: str
    retriever_counts: dict[str, int] = Field(default_factory=dict)
    candidate_count: int = 0
    final_count: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
