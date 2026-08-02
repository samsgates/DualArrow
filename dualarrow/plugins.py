from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

IngestionViewBuilder = Callable[[Any, str, dict[str, Any]], list[Any]]
RetrieverFactory = Callable[[Any, Any, dict[str, Any]], Any]
StoreFactory = Callable[[Any, Any], Any]

_ingestion_views: dict[str, IngestionViewBuilder] = {}
_retrievers: dict[str, RetrieverFactory] = {}
_store_factories: dict[str, StoreFactory] = {}
_loaded = False


def register_ingestion_view(name: str, builder: IngestionViewBuilder) -> None:
    if name in _ingestion_views:
        raise ValueError(f"Ingestion view already registered: {name}")
    _ingestion_views[name] = builder


def register_retriever(name: str, factory: RetrieverFactory) -> None:
    if name in _retrievers:
        raise ValueError(f"Retriever already registered: {name}")
    _retrievers[name] = factory


def register_store_provider(name: str, factory: StoreFactory) -> None:
    if name in _store_factories:
        raise ValueError(f"Store provider already registered: {name}")
    _store_factories[name] = factory


def load_entrypoint_plugins() -> None:
    global _loaded
    if _loaded:
        return
    groups = {
        "dualarrow.ingestion_views": register_ingestion_view,
        "dualarrow.retrievers": register_retriever,
        "dualarrow.store_providers": register_store_provider,
    }
    for group, register in groups.items():
        for entrypoint in entry_points(group=group):
            register(entrypoint.name, entrypoint.load())
    _loaded = True


def get_ingestion_view(name: str) -> IngestionViewBuilder | None:
    load_entrypoint_plugins()
    return _ingestion_views.get(name)


def get_retriever_factory(name: str) -> RetrieverFactory | None:
    load_entrypoint_plugins()
    return _retrievers.get(name)


def get_store_factory(name: str) -> StoreFactory | None:
    load_entrypoint_plugins()
    return _store_factories.get(name)
