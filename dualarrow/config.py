from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from dualarrow.errors import ConfigurationError
from dualarrow.models import StrictModel

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


class ArrowMDConfig(StrictModel):
    root: Path = Path("./knowledge")
    mode: Literal["strict_plain", "local_sidecar", "replicated"] = "local_sidecar"
    sidecar_path: Path | None = None
    include: list[str] = Field(default_factory=lambda: ["**/*.md"])
    exclude: list[str] = Field(default_factory=lambda: ["**/.git/**", "**/.dualarrow/**"])
    watch: bool = False

    @model_validator(mode="after")
    def fill_sidecar(self) -> "ArrowMDConfig":
        if self.sidecar_path is None:
            self.sidecar_path = self.root / ".dualarrow" / "dualarrow.db"
        return self


class StoreSpec(StrictModel):
    provider: str
    enabled: bool = True
    connection: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class ViewSpec(StrictModel):
    id: str
    type: str
    enabled: bool = True
    options: dict[str, Any] = Field(default_factory=dict)


class IngestionConfig(StrictModel):
    incremental: bool = True
    delete_removed_documents: bool = True
    quarantine_prompt_injection: bool = True
    views: list[ViewSpec] = Field(
        default_factory=lambda: [
            ViewSpec(id="full_document", type="full_document"),
            ViewSpec(
                id="markdown_sections",
                type="markdown_heading",
                options={"maximum_tokens": 1000},
            ),
            ViewSpec(
                id="parent_child",
                type="parent_child",
                options={"child_tokens": 180, "parent_tokens": 900},
            ),
        ]
    )


class EmbeddingConfig(StrictModel):
    provider: Literal["hash", "openai_compatible", "sentence_transformer"] = "hash"
    model: str = "dualarrow-hash-v1"
    dimensions: int = Field(default=384, ge=8, le=65536)
    base_url: str | None = None
    api_key: str | None = None
    batch_size: int = Field(default=64, ge=1, le=2048)


class GeneratorConfig(StrictModel):
    provider: Literal["extractive", "openai_compatible"] = "extractive"
    model: str = "extractive"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = Field(default=0.0, ge=0, le=2)
    timeout_seconds: float = Field(default=60, gt=0)


class RetrieverSpec(StrictModel):
    type: str
    enabled: bool = True
    top_k: int = Field(default=30, ge=1, le=1000)
    weight: float = Field(default=1.0, ge=0, le=100)
    options: dict[str, Any] = Field(default_factory=dict)


class RetrievalProfile(StrictModel):
    retrievers: list[RetrieverSpec]
    fusion: Literal["rrf", "weighted_rrf"] = "weighted_rrf"
    rrf_k: int = Field(default=60, ge=1)
    candidate_limit: int = Field(default=100, ge=1, le=5000)
    final_limit: int = Field(default=12, ge=1, le=200)
    reranker: Literal["none", "heuristic", "cross_encoder"] = "heuristic"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class VerificationConfig(StrictModel):
    enabled: bool = True
    minimum_claim_coverage: float = Field(default=0.8, ge=0, le=1)
    minimum_confidence: float = Field(default=0.72, ge=0, le=1)
    maximum_conflict_score: float = Field(default=0.25, ge=0, le=1)
    minimum_support_score: float = Field(default=0.46, ge=0, le=1)
    require_citations: bool = True
    require_multiple_sources: bool = False
    validate_numbers: bool = True


class SecurityConfig(StrictModel):
    api_key: str | None = None
    scan_prompt_injection: bool = True
    allowed_source_statuses: list[str] = Field(default_factory=lambda: ["approved"])


class APIConfig(StrictModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8080, ge=1, le=65535)
    workers: int = Field(default=1, ge=1, le=128)


class ProjectConfig(StrictModel):
    name: str = "DualArrow"
    domain: str = "general"
    risk_level: Literal["low", "medium", "high"] = "medium"
    default_language: str = "en"


class DualArrowConfig(StrictModel):
    schema_version: str = "1.0"
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    knowledge_store: ArrowMDConfig = Field(default_factory=ArrowMDConfig)
    stores: dict[str, StoreSpec] = Field(default_factory=dict)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    generation: GeneratorConfig = Field(default_factory=GeneratorConfig)
    retrieval_profiles: dict[str, RetrievalProfile] = Field(default_factory=dict)
    default_profile: str = "balanced"
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    @model_validator(mode="after")
    def defaults_and_validation(self) -> "DualArrowConfig":
        if not self.retrieval_profiles:
            self.retrieval_profiles = default_profiles()
        if self.default_profile not in self.retrieval_profiles:
            raise ValueError(f"Unknown default_profile: {self.default_profile}")
        return self


def default_profiles() -> dict[str, RetrievalProfile]:
    return {
        "fast": RetrievalProfile(
            retrievers=[
                RetrieverSpec(type="exact", top_k=20, weight=1.3),
                RetrieverSpec(type="bm25", top_k=40, weight=1.0),
                RetrieverSpec(type="dense", top_k=40, weight=1.0),
            ],
            final_limit=8,
            reranker="none",
        ),
        "balanced": RetrievalProfile(
            retrievers=[
                RetrieverSpec(type="exact", top_k=30, weight=1.4),
                RetrieverSpec(type="bm25", top_k=60, weight=1.0),
                RetrieverSpec(type="dense", top_k=60, weight=1.0),
                RetrieverSpec(type="graph", top_k=20, weight=0.6),
            ],
            candidate_limit=100,
            final_limit=12,
            reranker="heuristic",
        ),
        "verified_high_risk": RetrievalProfile(
            retrievers=[
                RetrieverSpec(type="exact", top_k=50, weight=1.6),
                RetrieverSpec(type="bm25", top_k=100, weight=1.2),
                RetrieverSpec(type="dense", top_k=100, weight=1.0),
                RetrieverSpec(type="graph", top_k=50, weight=0.8),
            ],
            candidate_limit=180,
            final_limit=20,
            reranker="heuristic",
        ),
    }


def _substitute_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        key, default = match.group(1), match.group(2)
        resolved = os.getenv(key, default)
        if resolved is None:
            raise ConfigurationError(f"Required environment variable is not set: {key}")
        return resolved

    return _ENV_PATTERN.sub(replace, value)


def load_config(path: str | Path) -> DualArrowConfig:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        raw = _substitute_env(raw)
        config = DualArrowConfig.model_validate(raw)
    except Exception as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(f"Invalid configuration {config_path}: {exc}") from exc

    base = config_path.parent
    if not config.knowledge_store.root.is_absolute():
        config.knowledge_store.root = (base / config.knowledge_store.root).resolve()
    if config.knowledge_store.sidecar_path and not config.knowledge_store.sidecar_path.is_absolute():
        config.knowledge_store.sidecar_path = (
            base / config.knowledge_store.sidecar_path
        ).resolve()
    return config
