from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    knowledge = root / "knowledge"
    knowledge.mkdir(parents=True)
    (knowledge / "machine.md").write_text(
        """---
id: machine.test
name: Test Machine
title: Test Machine Manual
status: approved
authority_score: 0.95
---
# Test Machine Manual
## Approved part
The approved bearing for machine AX-100 is BR-900.
## Limit
Stop AX-100 when vibration exceeds 8.0 millimetres per second RMS.
""",
        encoding="utf-8",
    )
    (root / "dualarrow.yaml").write_text(
        """schema_version: '1.0'
project:
  name: Test
  domain: manufacturing
  risk_level: medium
knowledge_store:
  root: ./knowledge
  sidecar_path: ./knowledge/.dualarrow/test.db
embedding:
  provider: hash
  dimensions: 128
generation:
  provider: extractive
verification:
  minimum_claim_coverage: 0.5
  minimum_confidence: 0.35
  minimum_support_score: 0.35
""",
        encoding="utf-8",
    )
    return root
