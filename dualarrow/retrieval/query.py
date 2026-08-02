from __future__ import annotations

import re
from dataclasses import dataclass, field

from dualarrow.config import DualArrowConfig
from dualarrow.models import QueryRequest

_IDENTIFIER_PATTERNS = [
    re.compile(r"\b[A-Z]{1,8}-\d{2,10}[A-Z0-9-]*\b"),
    re.compile(r"\b[A-Z]{2,8}\d{2,10}\b"),
    re.compile(r"\b(?:ICD|CPT|HCPCS)[-: ]?[A-Z0-9.]+\b", re.IGNORECASE),
]


@dataclass(slots=True)
class QueryAnalysis:
    normalized_query: str
    exact_identifiers: list[str] = field(default_factory=list)
    intent: str = "semantic_fact"
    risk_level: str = "medium"
    graph_required: bool = False
    numerical: bool = False
    temporal: bool = False


def analyze_query(request: QueryRequest, config: DualArrowConfig) -> QueryAnalysis:
    normalized = " ".join(request.query.strip().split())
    identifiers: list[str] = []
    for pattern in _IDENTIFIER_PATTERNS:
        identifiers.extend(pattern.findall(normalized))
    lowered = normalized.casefold()
    graph_words = {"why", "cause", "relationship", "connected", "depends", "impact", "root cause"}
    numerical = bool(re.search(r"\d", normalized)) or any(
        word in lowered for word in ("calculate", "percentage", "average", "total", "difference")
    )
    temporal = any(
        word in lowered for word in ("current", "latest", "before", "after", "between", "version")
    )
    if identifiers:
        intent = "exact_lookup"
    elif any(word in lowered for word in ("why", "root cause", "caused", "failure")):
        intent = "root_cause"
    elif any(word in lowered for word in ("compare", "difference", "versus", " vs ")):
        intent = "comparison"
    elif numerical:
        intent = "numerical"
    else:
        intent = "semantic_fact"
    return QueryAnalysis(
        normalized_query=normalized,
        exact_identifiers=sorted(set(identifiers)),
        intent=intent,
        risk_level=config.project.risk_level,
        graph_required=any(word in lowered for word in graph_words),
        numerical=numerical,
        temporal=temporal,
    )


def select_profile(
    request: QueryRequest, analysis: QueryAnalysis, config: DualArrowConfig
) -> str:
    if request.retrieval_profile != "auto":
        if request.retrieval_profile not in config.retrieval_profiles:
            raise ValueError(f"Unknown retrieval profile: {request.retrieval_profile}")
        return request.retrieval_profile
    if config.project.risk_level == "high" and "verified_high_risk" in config.retrieval_profiles:
        return "verified_high_risk"
    return config.default_profile
