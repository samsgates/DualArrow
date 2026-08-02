# Release Validation

Release: DualArrow 0.1.0

Validated on Python 3.13 in the build environment.

## Completed checks

- 15 automated tests passed.
- Test coverage was 76 percent, above the configured 65 percent quality gate.
- The Python wheel was built successfully with setuptools.
- The installed wheel completed an independent quickstart smoke test.
- The quickstart configuration passed schema validation.
- Local ArrowMD ingestion and verified querying completed successfully.
- Incremental ingestion, deletion, prompt-injection quarantine, role filtering, numeric mismatch detection, API health, API querying, evaluation metrics, plugin registration, and strict plain mode are covered by automated tests.
- Source files were scanned for common embedded private-key and cloud-key patterns. No matches were found.

## Environment limitation

Live Qdrant, OpenSearch, Neo4j, GraphDB, and pgvector services were not available in the build environment. Their adapters are included, importable, configuration-validated, and isolated behind provider contracts, but should be exercised against the target service versions before a production deployment.

## Recommended release commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
coverage run -m pytest
coverage report --fail-under=65
python -m pip wheel . --no-deps --no-build-isolation
```
