from __future__ import annotations

from pathlib import Path

from dualarrow.config import DualArrowConfig, load_config
from dualarrow.embeddings.factory import create_embedding_provider
from dualarrow.evidence import EvidenceLedger
from dualarrow.generation.factory import create_generator
from dualarrow.ingestion.pipeline import IngestionPipeline
from dualarrow.models import AnswerResult, QueryRequest
from dualarrow.observability.tracing import set_trace_id
from dualarrow.retrieval.engine import RetrievalEngine
from dualarrow.stores.factory import create_store
from dualarrow.verification.verifier import EvidenceVerifier


class DualArrow:
    def __init__(self, config: DualArrowConfig) -> None:
        self.config = config
        self.embeddings = create_embedding_provider(config.embedding)
        self.store = create_store(config)
        self.ingestion = IngestionPipeline(config, self.store, self.embeddings)
        self.retrieval = RetrievalEngine(config, self.store, self.embeddings)
        self.generator = create_generator(config.generation)
        self.verifier = EvidenceVerifier(config.verification)
        canonical_path = self.store.canonical.path
        self.ledger = EvidenceLedger(
            canonical_path.with_name(canonical_path.stem + "-evidence.db"),
            ephemeral=config.knowledge_store.mode == "strict_plain",
        )
        self._started = False

    @classmethod
    def from_config(cls, path: str | Path) -> "DualArrow":
        return cls(load_config(path))

    async def start(self) -> None:
        if self._started:
            return
        await self.store.initialize()
        await self.ledger.initialize()
        self._started = True

    async def close(self) -> None:
        if not self._started:
            return
        await self.store.close()
        await self.ledger.close()
        for component in (self.embeddings, self.generator):
            close = getattr(component, "close", None)
            if close:
                await close()
        self._started = False

    async def ingest(self, force: bool = False):
        await self.start()
        return await self.ingestion.run(force=force)

    async def query(self, request: QueryRequest | str) -> AnswerResult:
        await self.start()
        if isinstance(request, str):
            request = QueryRequest(query=request)
        evidence, trace = await self.retrieval.retrieve(request)
        set_trace_id(trace.trace_id)
        draft = await self.generator.generate(request.query, evidence)
        outcome = await self.verifier.verify(request.query, draft, evidence)
        answer = AnswerResult(
            status=outcome.status,
            answer=outcome.answer,
            claims=outcome.claims,
            citations=outcome.citations,
            evidence=evidence if request.include_evidence else [],
            confidence=outcome.confidence,
            trace_id=trace.trace_id,
            retrieval_profile=trace.selected_profile,
            warnings=outcome.warnings,
        )
        await self.ledger.record(trace, answer)
        return answer

    async def __aenter__(self) -> "DualArrow":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
