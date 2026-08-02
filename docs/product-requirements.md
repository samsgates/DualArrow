# Product requirements

## Product

DualArrow is an evidence-first adaptive RAG framework. It supports plain Markdown, specialized databases, configurable ingestion views, hybrid retrieval, graph retrieval, evidence verification, and abstention.

## Required outcomes

1. Developers can start with one Markdown directory.
2. The same application can switch to Qdrant, pgvector, OpenSearch, Neo4j, or GraphDB through configuration.
3. Multiple retrievers can run in parallel and be fused.
4. Every answer claim can be traced to source evidence.
5. Unsupported, contradictory, stale, or unauthorized evidence can be rejected.
6. Evaluation can measure retrieval and answer quality separately.
7. Custom retrieval and ingestion methods can be installed as plugins.

## v0.1 acceptance criteria

- Incremental Markdown ingestion works.
- Deleted and quarantined files disappear from retrieval.
- Exact, BM25, dense, and graph retrieval work in local mode.
- External adapter packages build and import through optional extras.
- Verified responses include claims, citations, evidence, confidence, status, and trace ID.
- The API and CLI execute the same pipeline.
- Tests cover lifecycle, authorization, injection quarantine, numeric mismatch, evaluation, and packaging.

## Non-goals for v0.1

- Autonomous clinical or industrial decision making
- Guaranteed elimination of hallucinations
- Full visual document retrieval
- Distributed transaction guarantees across every external database
- A hosted SaaS control plane
