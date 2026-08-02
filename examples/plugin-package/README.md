# DualArrow example plugin

Install this directory in editable mode:

```bash
pip install -e examples/plugin-package
```

Use the ingestion view:

```yaml
ingestion:
  views:
    - id: first_paragraphs
      type: first_paragraph
      options:
        maximum_characters: 1000
```

Use the retriever:

```yaml
retrieval_profiles:
  trusted:
    retrievers:
      - type: authority_bm25
        top_k: 30
        weight: 1.0
        options:
          minimum_authority: 0.95
    final_limit: 10
    reranker: heuristic
```
