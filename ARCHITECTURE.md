# DualArrow Architecture

The complete architecture guide is in [docs/architecture.md](docs/architecture.md).

DualArrow separates evidence discovery from evidence assurance:

```text
Arrow One: query analysis -> multi-method retrieval -> fusion -> reranking
Arrow Two: claim extraction -> evidence mapping -> verification -> abstention
```

Canonical evidence remains reconstructable from every derived index.
