from __future__ import annotations

from typing import Any

import os
import tempfile
from pathlib import Path

from dualarrow.config import DualArrowConfig, StoreSpec
from dualarrow.errors import ConfigurationError
from dualarrow.stores.arrowmd import ArrowMDStore
from dualarrow.stores.composite import CompositeKnowledgeStore
from dualarrow.stores.graphdb import GraphDBAdapter
from dualarrow.stores.neo4j import Neo4jGraphAdapter
from dualarrow.stores.opensearch import OpenSearchLexicalAdapter
from dualarrow.stores.pgvector import PgVectorAdapter
from dualarrow.stores.qdrant import QdrantVectorAdapter
from dualarrow.plugins import get_store_factory
from dualarrow.utils import stable_id


def _required(spec: StoreSpec, key: str) -> Any:
    value = spec.options.get(key)
    if value is None:
        raise ConfigurationError(f"Store {spec.provider} requires options.{key}")
    return value


def create_store(config: DualArrowConfig) -> CompositeKnowledgeStore:
    sidecar = config.knowledge_store.sidecar_path
    if sidecar is None:
        raise ConfigurationError("knowledge_store.sidecar_path is required")
    ephemeral = config.knowledge_store.mode == "strict_plain"
    if ephemeral:
        temp_root = Path(tempfile.gettempdir()) / "dualarrow"
        sidecar = temp_root / f"{stable_id(str(config.knowledge_store.root))}-{os.getpid()}.db"
    canonical = ArrowMDStore(sidecar, config.embedding.dimensions, ephemeral=ephemeral)
    vector = None
    lexical = None
    graph = None

    for role, spec in config.stores.items():
        if not spec.enabled:
            continue
        provider = spec.provider.casefold()
        if role == "vector":
            if provider == "qdrant":
                vector = QdrantVectorAdapter(
                    url=spec.connection or "http://localhost:6333",
                    collection=spec.options.get("collection", "dualarrow"),
                    dimensions=config.embedding.dimensions,
                    api_key=spec.options.get("api_key"),
                    prefer_grpc=bool(spec.options.get("prefer_grpc", False)),
                )
            elif provider in {"pgvector", "postgresql"}:
                if not spec.connection:
                    raise ConfigurationError("pgvector store requires connection")
                vector = PgVectorAdapter(
                    connection=spec.connection,
                    table=spec.options.get("table", "dualarrow_units"),
                    dimensions=config.embedding.dimensions,
                )
            elif provider not in {"arrowmd", "local"}:
                plugin = get_store_factory(provider)
                if plugin is None:
                    raise ConfigurationError(f"Unsupported vector provider: {provider}")
                vector = plugin(spec, config)
        elif role == "lexical":
            if provider == "opensearch":
                lexical = OpenSearchLexicalAdapter(
                    url=spec.connection or "http://localhost:9200",
                    index=spec.options.get("index", "dualarrow"),
                    username=spec.options.get("username"),
                    password=spec.options.get("password"),
                    verify_certs=bool(spec.options.get("verify_certs", True)),
                )
            elif provider not in {"arrowmd", "local"}:
                plugin = get_store_factory(provider)
                if plugin is None:
                    raise ConfigurationError(f"Unsupported lexical provider: {provider}")
                lexical = plugin(spec, config)
        elif role in {"graph", "rdf"}:
            if provider == "neo4j":
                graph = Neo4jGraphAdapter(
                    uri=spec.connection or "neo4j://localhost:7687",
                    username=_required(spec, "username"),
                    password=_required(spec, "password"),
                    database=spec.options.get("database", "neo4j"),
                )
            elif provider == "graphdb":
                graph = GraphDBAdapter(
                    query_endpoint=_required(spec, "query_endpoint"),
                    update_endpoint=_required(spec, "update_endpoint"),
                    username=spec.options.get("username"),
                    password=spec.options.get("password"),
                )
            elif provider not in {"arrowmd", "local"}:
                plugin = get_store_factory(provider)
                if plugin is None:
                    raise ConfigurationError(f"Unsupported graph provider: {provider}")
                graph = plugin(spec, config)

    return CompositeKnowledgeStore(canonical, vector=vector, lexical=lexical, graph=graph)
