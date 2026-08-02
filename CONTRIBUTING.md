# Contributing to DualArrow

Thank you for helping improve evidence-first RAG.

1. Open an issue for substantial features or interface changes.
2. Create a focused branch.
3. Add tests for every behavior change.
4. Run `ruff check .`, `mypy dualarrow`, and `pytest`.
5. Update documentation and the changelog.

Database adapters must pass the adapter conformance suite and preserve canonical evidence provenance. Generated summaries, graph edges, and propositions must never become authoritative without a source span.
