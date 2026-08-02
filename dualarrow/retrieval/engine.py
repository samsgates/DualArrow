from __future__ import annotations

import asyncio
import time

from dualarrow.config import DualArrowConfig, RetrievalProfile
from dualarrow.embeddings.base import EmbeddingProvider
from dualarrow.models import EvidenceCandidate, QueryRequest, RetrievalTrace
from dualarrow.plugins import get_retriever_factory
from dualarrow.retrieval.fusion import weighted_rrf
from dualarrow.retrieval.query import analyze_query, select_profile
from dualarrow.retrieval.rerank import CrossEncoderReranker, HeuristicReranker
from dualarrow.retrieval.retrievers import BM25Retriever, DenseRetriever, ExactRetriever, GraphRetriever
from dualarrow.stores.base import KnowledgeStore
from dualarrow.utils import new_id


class RetrievalEngine:
    def __init__(
        self,
        config: DualArrowConfig,
        store: KnowledgeStore,
        embeddings: EmbeddingProvider,
    ) -> None:
        self.config = config
        self.store = store
        self.embeddings = embeddings
        self.retrievers = {
            "exact": ExactRetriever(store),
            "bm25": BM25Retriever(store),
            "dense": DenseRetriever(store, embeddings),
            "graph": GraphRetriever(store),
        }
        self._cross_encoders: dict[str, CrossEncoderReranker] = {}

    async def retrieve(
        self, request: QueryRequest
    ) -> tuple[list[EvidenceCandidate], RetrievalTrace]:
        started = time.perf_counter()
        analysis = analyze_query(request, self.config)
        profile_name = select_profile(request, analysis, self.config)
        profile = self.config.retrieval_profiles[profile_name]
        tasks = []
        active_specs = []
        for spec in profile.retrievers:
            if not spec.enabled:
                continue
            retriever = self.retrievers.get(spec.type)
            if retriever is None:
                factory = get_retriever_factory(spec.type)
                if factory is None:
                    raise ValueError(f"Unknown retriever type: {spec.type}")
                retriever = factory(self.store, self.embeddings, spec.options)
                self.retrievers[spec.type] = retriever
            active_specs.append(spec)
            tasks.append(
                asyncio.create_task(
                    retriever.retrieve(
                        analysis.normalized_query,
                        spec.top_k,
                        request.metadata_filters,
                    ),
                    name=f"retriever:{spec.type}",
                )
            )
        settled = await asyncio.gather(*tasks, return_exceptions=True)
        result_sets = []
        counts = {}
        errors = []
        for spec, result in zip(active_specs, settled, strict=True):
            if isinstance(result, BaseException):
                errors.append(f"{spec.type}: {result}")
                candidates = []
            else:
                candidates = [
                    candidate
                    for candidate in result
                    if self._candidate_allowed(candidate, request)
                ]
            counts[spec.type] = len(candidates)
            result_sets.append((spec.type, spec.weight, candidates))
        fused = weighted_rrf(
            result_sets,
            rrf_k=profile.rrf_k,
            limit=profile.candidate_limit,
        )
        final = await self._rerank(analysis.normalized_query, fused, profile)
        trace = RetrievalTrace(
            trace_id=new_id("trace"),
            query=request.query,
            normalized_query=analysis.normalized_query,
            selected_profile=profile_name,
            retriever_counts=counts,
            candidate_count=len(fused),
            final_count=len(final),
            timings_ms={"total": (time.perf_counter() - started) * 1000},
            metadata={
                "intent": analysis.intent,
                "exact_identifiers": analysis.exact_identifiers,
                "retriever_errors": errors,
            },
        )
        return final, trace


    def _candidate_allowed(self, candidate: EvidenceCandidate, request: QueryRequest) -> bool:
        metadata = candidate.metadata
        status = str(metadata.get("status", "approved"))
        if status not in self.config.security.allowed_source_statuses:
            return False
        tenant = metadata.get("tenant_id")
        if tenant is not None and str(tenant) != request.tenant_id:
            return False
        access = metadata.get("access") if isinstance(metadata.get("access"), dict) else {}
        required_roles = set(access.get("roles") or [])
        if required_roles and not required_roles.intersection(request.user_roles):
            return False
        for key, expected in request.metadata_filters.items():
            current = metadata
            for part in key.split("."):
                if not isinstance(current, dict) or part not in current:
                    return False
                current = current[part]
            if isinstance(expected, list):
                if current not in expected and not (isinstance(current, list) and set(current).intersection(expected)):
                    return False
            elif isinstance(current, list):
                if expected not in current:
                    return False
            elif current != expected:
                return False
        return True

    async def _rerank(
        self,
        query: str,
        candidates: list[EvidenceCandidate],
        profile: RetrievalProfile,
    ) -> list[EvidenceCandidate]:
        if profile.reranker == "none":
            return candidates[: profile.final_limit]
        if profile.reranker == "heuristic":
            return HeuristicReranker().rerank(query, candidates, profile.final_limit)
        reranker = self._cross_encoders.get(profile.reranker_model)
        if reranker is None:
            reranker = CrossEncoderReranker(profile.reranker_model)
            self._cross_encoders[profile.reranker_model] = reranker
        return await reranker.rerank(query, candidates, profile.final_limit)
