# DualArrow

**Retrieve broadly. Verify precisely.**

DualArrow is an evidence-first, database-agnostic Retrieval-Augmented Generation framework. It combines configurable ingestion, hybrid retrieval, graph retrieval, evidence fusion, claim-level verification, citation validation, conflict detection, confidence scoring, and safe abstention.

DualArrow is designed for projects where an answer must be supported by traceable evidence, including medical knowledge, mathematics, science, manufacturing, engineering, policy, compliance, and enterprise documentation.

> DualArrow reduces hallucination risk. It cannot guarantee that any language model will never make a mistake. High-risk outputs still require appropriate expert review.

## Why DualArrow

Most RAG systems follow one path:

```text
chunk -> embed -> vector search -> prompt -> answer
```

DualArrow separates the work into two independent arrows:

```text
Arrow One: discover the strongest available evidence
Arrow Two: verify that the evidence supports every material claim
```

A verified answer retains this chain:

```text
answer -> claim -> evidence -> source span -> document version -> canonical source
```

## Main features

- Plain Markdown knowledge store through **ArrowMD**
- Local zero-infrastructure mode with SQLite FTS5 and file-backed vectors
- Multiple ingestion views for the same source
- Exact, BM25, dense, and graph retrieval
- Configurable parallel retrieval and weighted Reciprocal Rank Fusion
- Optional cross-encoder reranking
- Query-aware retrieval profiles
- Evidence-constrained extractive or LLM generation
- Atomic claim verification
- Number and negation validation
- Citation validation and evidence provenance
- Conflict detection
- Confidence breakdown and safe abstention
- Qdrant, pgvector, OpenSearch, Neo4j, and GraphDB adapters
- FastAPI service and Python API
- Typer CLI
- Evaluation runner and CI quality gates
- Docker and Docker Compose deployment examples

## Supported storage combinations

DualArrow always preserves a canonical local evidence representation. Optional specialized databases can be selected by role.

| Role | Supported production adapter |
|---|---|
| Canonical Markdown | ArrowMD |
| Local lexical search | SQLite FTS5 |
| Local dense search | File-backed NumPy vectors |
| Vector database | Qdrant, pgvector |
| Lexical database | OpenSearch |
| Property graph | Neo4j |
| RDF and SPARQL | GraphDB |
| Object storage | Filesystem in v0.1, S3-compatible connector planned |

The adapter contracts are designed for Milvus, Weaviate, Pinecone, Elasticsearch, MongoDB, ArangoDB, Neptune, TimescaleDB, Redis, Oracle, and other community providers.

## Retrieval methods included

- Exact substring retrieval
- Identifier-oriented retrieval
- BM25 retrieval
- Dense vector retrieval
- Markdown link graph retrieval
- Neo4j graph expansion
- GraphDB SPARQL graph retrieval
- Weighted Reciprocal Rank Fusion
- Heuristic evidence-aware reranking
- Optional cross-encoder reranking
- Metadata filters
- Parent-child context
- Sentence-window context
- Atomic proposition view

## Quickstart

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install DualArrow

From the repository:

```bash
pip install -e .
```

For local sentence-transformer embeddings and cross-encoder reranking:

```bash
pip install -e ".[local-ml]"
```

For all maintained database adapters:

```bash
pip install -e ".[all-adapters]"
```

### 3. Initialize a project

```bash
dualarrow init my-dualarrow-project
cd my-dualarrow-project
```

The generated project contains:

```text
my-dualarrow-project/
├── dualarrow.yaml
└── knowledge/
    ├── machines/p104.md
    ├── parts/br-442.md
    ├── science/evidence-guide.md
    └── standards/press-safety.md
```

### 4. Ingest the knowledge

```bash
dualarrow ingest --config dualarrow.yaml
```

### 5. Ask a question

```bash
dualarrow query "Which bearing is approved for P-104?" --config dualarrow.yaml
```

The response includes:

- Answer status
- Verified answer
- Atomic claims
- Claim support scores
- Citations
- Retrieved evidence
- Confidence breakdown
- Trace ID

### 6. Start the API

```bash
dualarrow serve --config dualarrow.yaml
```

Open:

- API: `http://localhost:8080`
- OpenAPI: `http://localhost:8080/docs`
- Health: `http://localhost:8080/health`

Query the API:

```bash
curl -X POST http://localhost:8080/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the P-104 bearing replacement interval?",
    "retrieval_profile": "auto",
    "include_evidence": true
  }'
```

## Python API

```python
import asyncio

from dualarrow import DualArrow, QueryRequest


async def main() -> None:
    async with DualArrow.from_config("dualarrow.yaml") as engine:
        report = await engine.ingest()
        print(report)

        result = await engine.query(
            QueryRequest(
                query="Which bearing is approved for P-104?",
                retrieval_profile="balanced",
                include_evidence=True,
            )
        )
        print(result.model_dump_json(indent=2))


asyncio.run(main())
```

## ArrowMD

ArrowMD treats ordinary Markdown files as a canonical knowledge database.

```markdown
---
id: procedure.press-p104.bearing-replacement
title: Press P-104 Bearing Replacement
status: approved
authority_score: 0.98
version: "3.2"
effective_from: 2026-05-01
tags: [bearing, maintenance]
relations:
  - type: applies_to
    target: machine.press-p104
---

# Press P-104 Bearing Replacement

## Required part

The approved bearing is BR-442.
```

ArrowMD supports:

- YAML frontmatter
- Heading hierarchy
- Markdown links and wiki-style links
- Tables
- Code blocks
- Equations
- Images and references
- Stable document IDs
- File and line citations
- Content hashes
- Git-friendly versioning

### ArrowMD modes

| Mode | Description |
|---|---|
| `strict_plain` | Markdown is the only persistent knowledge format |
| `local_sidecar` | Markdown is canonical, with a local `.dualarrow` SQLite sidecar |
| `replicated` | Markdown is canonical and indexes are replicated to external databases |

In `strict_plain` mode, DualArrow creates an ephemeral index in the operating system temporary directory and removes it on shutdown. Only Markdown remains in the project knowledge directory. The process must ingest again after restart.

## PolyView ingestion

One source can produce several retrieval views:

```yaml
 ingestion:
   views:
     - id: full_document
       type: full_document

     - id: markdown_sections
       type: markdown_heading
       options:
         maximum_tokens: 1000

     - id: recursive_chunks
       type: recursive
       options:
         chunk_size: 500
         overlap: 75

     - id: parent_child
       type: parent_child
       options:
         child_tokens: 180
         parent_tokens: 900

     - id: propositions
       type: atomic_propositions
```

Generated summaries or propositions must retain original source references. The built-in proposition view uses verbatim sentences and does not invent generated facts.

Additional built-in views include:

- `table`. Preserves complete Markdown tables and creates row-level facts
- `equation`. Extracts inline and block LaTeX expressions
- `code`. Extracts fenced code blocks with language metadata
- `hierarchy`. Creates heading-path navigation units

Custom views and retrievers can be installed through Python entry points. See `examples/plugin-package`.

## Retrieval profiles

Retrieval profiles define which methods run and how they are combined.

```yaml
retrieval_profiles:
  balanced:
    retrievers:
      - {type: exact, top_k: 30, weight: 1.5}
      - {type: bm25, top_k: 60, weight: 1.1}
      - {type: dense, top_k: 60, weight: 1.0}
      - {type: graph, top_k: 20, weight: 0.6}
    fusion: weighted_rrf
    candidate_limit: 100
    final_limit: 12
    reranker: heuristic
```

Built-in profile names:

- `fast`
- `balanced`
- `verified_high_risk`

Use a profile explicitly:

```bash
dualarrow query "Why can BR-442 overheat?" --profile verified_high_risk
```

## Embedding providers

### Hash provider

```yaml
embedding:
  provider: hash
  dimensions: 384
```

This deterministic provider requires no model download or network request. It is useful for tests, demonstrations, and constrained local deployments. It is not recommended as the final embedding model for accuracy-sensitive production systems.

### Sentence Transformers

```yaml
embedding:
  provider: sentence_transformer
  model: BAAI/bge-m3
  dimensions: 1024
```

Install:

```bash
pip install -e ".[local-ml]"
```

### OpenAI-compatible embeddings

```yaml
embedding:
  provider: openai_compatible
  model: ${EMBEDDING_MODEL}
  dimensions: 1024
  base_url: ${EMBEDDING_BASE_URL}
  api_key: ${EMBEDDING_API_KEY}
```

The endpoint must implement the OpenAI-compatible `/embeddings` API.

## Answer generators

### Extractive generator

```yaml
generation:
  provider: extractive
```

Extractive generation selects sentences directly from evidence. This is the safest zero-network option and is useful for validating the retrieval and verification pipeline.

### OpenAI-compatible generator

```yaml
generation:
  provider: openai_compatible
  model: ${GENERATION_MODEL}
  base_url: ${GENERATION_BASE_URL}
  api_key: ${GENERATION_API_KEY}
  temperature: 0
```

The endpoint must implement `/chat/completions` and JSON output.

## Verification

```yaml
verification:
  minimum_claim_coverage: 0.95
  minimum_confidence: 0.85
  maximum_conflict_score: 0.20
  minimum_support_score: 0.55
  require_citations: true
  require_multiple_sources: true
  validate_numbers: true
```

Possible response statuses:

- `VERIFIED_ANSWER`
- `VERIFIED_PARTIAL_ANSWER`
- `CONFLICTED_ANSWER`
- `CLARIFICATION_REQUIRED`
- `INSUFFICIENT_EVIDENCE`
- `UNAUTHORIZED`
- `UNSAFE_SOURCE`
- `OUT_OF_SCOPE`
- `SYSTEM_ERROR`

The built-in verifier checks lexical support, numbers, negation mismatches, citation existence, source authority, freshness, evidence coverage, and configured multi-source requirements. A future verifier plugin can add domain NLI models, theorem provers, or external validation services.

## Qdrant

```yaml
stores:
  vector:
    provider: qdrant
    connection: http://localhost:6333
    options:
      collection: dualarrow
      api_key: ${QDRANT_API_KEY:-}
```

Install:

```bash
pip install -e ".[qdrant]"
```

## pgvector

```yaml
stores:
  vector:
    provider: pgvector
    connection: postgresql+asyncpg://dualarrow:password@localhost:5432/dualarrow
    options:
      table: dualarrow_units
```

Install:

```bash
pip install -e ".[postgres]"
```

## OpenSearch

```yaml
stores:
  lexical:
    provider: opensearch
    connection: https://localhost:9200
    options:
      index: dualarrow
      username: admin
      password: ${OPENSEARCH_PASSWORD}
      verify_certs: true
```

## Neo4j

```yaml
stores:
  graph:
    provider: neo4j
    connection: neo4j://localhost:7687
    options:
      username: neo4j
      password: ${NEO4J_PASSWORD}
      database: neo4j
```

## GraphDB

```yaml
stores:
  rdf:
    provider: graphdb
    options:
      query_endpoint: http://localhost:7200/repositories/dualarrow
      update_endpoint: http://localhost:7200/repositories/dualarrow/statements
      username: ${GRAPHDB_USERNAME:-}
      password: ${GRAPHDB_PASSWORD:-}
```

GraphDB installation and licensing are intentionally not bundled in Docker Compose. Connect DualArrow to an existing GraphDB repository.

## Multi-database production profile

See `examples/configs/production-multidb.yaml` for external model providers and `examples/configs/docker-multidb.yaml` for a fully local multi-database demonstration.

```text
ArrowMD       canonical evidence
Qdrant       dense vector retrieval
OpenSearch   lexical and identifier retrieval
Neo4j        graph expansion
SQLite       evidence ledger and local fallback
```

Start the example infrastructure:

```bash
cp .env.example .env
docker compose --profile production up --build
```

The Compose file pins Qdrant 1.18.2, OpenSearch 3.7.0, and Neo4j 2026.06.0 Community. Review and update pins through your normal dependency-management process before production deployment.

## Evaluation

Dataset format:

```json
{"query":"Which bearing is approved for P-104?","expected_answer":"The approved bearing is BR-442."}
```

Relevant unit IDs may also be included:

```json
{
  "query": "Which bearing is approved for P-104?",
  "relevant_unit_ids": ["unit_..."],
  "expected_answer": "The approved bearing is BR-442.",
  "profile": "balanced"
}
```

Run:

```bash
dualarrow evaluate examples/evaluation/sample.jsonl \
  --config examples/quickstart/dualarrow.yaml \
  --output evaluation-result.json
```

Metrics include:

- Precision@K
- Recall@K
- Mean Reciprocal Rank
- nDCG@K
- Answer token F1
- Average confidence

## Database advisor

```bash
dualarrow db recommend --chunks 500000 --graph --no-minimal-ops
```

The advisor provides a starting stack recommendation. Final selection should be based on a representative benchmark with your documents, filters, traffic, and accuracy requirements.

## CLI reference

```text
dualarrow init
dualarrow ingest
dualarrow query
dualarrow serve
dualarrow evaluate
dualarrow doctor
dualarrow config validate
dualarrow config explain
dualarrow db recommend
```


## Plugin development

DualArrow discovers plugins through these entry-point groups:

```text
dualarrow.ingestion_views
dualarrow.retrievers
dualarrow.store_providers
```

A plugin can add a custom ingestion view, retrieval algorithm, or database provider without modifying DualArrow core. The complete example package is available in `examples/plugin-package`.

## Security model

DualArrow includes:

- Prompt-injection scanning during ingestion
- Quarantine for high-severity patterns
- Source approval status
- Content hashes
- Stable provenance
- Pre-context metadata filters
- API-key authentication
- Secret redaction from the config endpoint
- Immutable query and answer traces

Production deployments should also use:

- TLS
- Network segmentation
- Managed secrets
- Database-native identity and permissions
- Tenant-specific namespaces or databases
- Read-only source credentials
- Backup and recovery procedures
- Security monitoring
- Human approval for high-risk actions

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
mypy dualarrow
```

Run coverage:

```bash
pytest --cov=dualarrow --cov-report=term-missing
```

## Use-case configuration examples

- `examples/configs/math-local.yaml`
- `examples/configs/science-local.yaml`
- `examples/configs/graphdb-rdf.yaml`
- `examples/configs/docker-multidb.yaml`
- `examples/configs/production-multidb.yaml`

Product requirements, architecture, production guidance, adapter development, evaluation guidance, and the roadmap are available under `docs/` and the repository root.

## Repository structure

```text
dualarrow/
├── dualarrow/
│   ├── api/
│   ├── embeddings/
│   ├── evaluation/
│   ├── generation/
│   ├── ingestion/
│   ├── retrieval/
│   ├── security/
│   ├── stores/
│   └── verification/
├── docs/
├── examples/
├── tests/
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Current implementation scope

Version 0.1 provides a complete, runnable evidence-first RAG foundation and maintained adapters for the primary database combinations described above.

The following PRD capabilities are extension points rather than fully implemented v0.1 runtime modules:

- Visual-page retrieval
- Table-specific neural retrieval
- Equation similarity indexes
- SQL and time-series query generation
- Learned router training
- Learned fusion
- Full tenant administration dashboard
- Milvus, Weaviate, Pinecone, MongoDB, ArangoDB, Neptune, and TimescaleDB adapters
- Distributed ingestion queues

The interfaces and configuration model are structured to add these features without changing application code.

## License

Apache License 2.0. See `LICENSE`.
