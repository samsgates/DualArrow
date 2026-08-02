# Adapter development

Adapters declare capabilities and implement one focused role.

- Vector adapters implement `VectorIndexAdapter`.
- Lexical adapters implement `LexicalIndexAdapter`.
- Graph and RDF adapters implement `GraphIndexAdapter`.
- Canonical stores implement `KnowledgeStore`.

Every returned candidate must contain enough provenance to reconstruct the canonical evidence. Do not return an embedding, graph summary, or generated proposition without its original document ID and evidence location.

A new adapter should include:

1. Optional dependency extra
2. Configuration validation
3. Health check
4. Idempotent document replacement
5. Safe deletion behavior
6. Metadata filters
7. Timeout and cancellation handling
8. Unit and integration tests
9. Documentation and sample configuration


## Entry-point plugins

Register external providers in `pyproject.toml`:

```toml
[project.entry-points."dualarrow.retrievers"]
my_retriever = "my_package:create_retriever"

[project.entry-points."dualarrow.ingestion_views"]
my_view = "my_package:build_view"

[project.entry-points."dualarrow.store_providers"]
my_database = "my_package:create_store"
```

See `examples/plugin-package` for an installable reference.
