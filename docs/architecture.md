# Architecture

## Two-arrow execution

```text
Query
  -> Query analysis
  -> Retrieval profile selection
  -> Parallel retrievers
  -> Weighted RRF
  -> Reranking
  -> Evidence candidates
  -> Evidence-constrained generation
  -> Atomic claims
  -> Claim and citation verification
  -> Confidence and abstention
```

## Canonical and derived knowledge

Original Markdown is canonical. Sections, chunks, embeddings, FTS rows, graph nodes, and RDF triples are derived. Every derived unit retains a document ID, source URI, line range, checksum, and view ID.

## Consistency model

The local canonical index is written before external indexes. External stores can be rebuilt from ArrowMD. For strict enterprise synchronization, extend `CompositeKnowledgeStore` with an outbox and knowledge-version coordinator.

## Failure behavior

Parallel retriever failures are isolated and recorded in the trace. The remaining retrievers can still produce an answer. In high-risk production environments, configure deployment monitoring to fail closed when a required adapter is unhealthy.
