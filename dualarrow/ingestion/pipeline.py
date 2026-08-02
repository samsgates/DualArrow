from __future__ import annotations

import fnmatch
import logging
from pathlib import Path

from dualarrow.config import DualArrowConfig
from dualarrow.embeddings.base import EmbeddingProvider
from dualarrow.ingestion.chunking import build_units
from dualarrow.ingestion.markdown import parse_markdown
from dualarrow.models import IngestionReport
from dualarrow.security.injection import scan_prompt_injection
from dualarrow.stores.base import KnowledgeStore
from dualarrow.utils import new_id, sha256_text, stable_id

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        config: DualArrowConfig,
        store: KnowledgeStore,
        embeddings: EmbeddingProvider,
    ) -> None:
        self.config = config
        self.store = store
        self.embeddings = embeddings

    def _discover(self) -> list[Path]:
        root = self.config.knowledge_store.root
        root.mkdir(parents=True, exist_ok=True)
        files: set[Path] = set()
        for pattern in self.config.knowledge_store.include:
            files.update(path for path in root.glob(pattern) if path.is_file())
        result = []
        for path in files:
            relative = path.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(relative, pattern) for pattern in self.config.knowledge_store.exclude):
                continue
            result.append(path)
        return sorted(result)

    async def run(self, force: bool = False) -> IngestionReport:
        run_id = new_id("ing")
        files = self._discover()
        seen_ids: set[str] = set()
        fingerprints: list[str] = []
        index_signature = sha256_text(
            self.config.ingestion.model_dump_json() + self.config.embedding.model_dump_json()
        )
        documents_indexed = 0
        units_created = 0
        quarantined = 0
        errors: list[str] = []

        for path in files:
            try:
                parsed = parse_markdown(path, self.config.knowledge_store.root)
                if parsed.document.status not in self.config.security.allowed_source_statuses:
                    logger.info("Skipping unapproved document: %s", path)
                    continue
                if (
                    self.config.security.scan_prompt_injection
                    and self.config.ingestion.quarantine_prompt_injection
                ):
                    findings = scan_prompt_injection(parsed.document.content)
                    if any(item.severity == "high" for item in findings):
                        quarantined += 1
                        logger.warning("Quarantined suspicious document: %s", path)
                        continue
                source_checksum = parsed.document.checksum
                effective_checksum = sha256_text(source_checksum + index_signature)
                parsed.document.metadata["source_checksum"] = source_checksum
                parsed.document.metadata["index_signature"] = index_signature
                parsed.document.checksum = effective_checksum
                seen_ids.add(parsed.document.document_id)
                fingerprints.append(f"{parsed.document.document_id}:{effective_checksum}")
                current = await self.store.document_checksum(parsed.document.document_id)
                if not force and self.config.ingestion.incremental and current == effective_checksum:
                    continue
                units = build_units(parsed, self.config.ingestion.views)
                vectors = await self.embeddings.embed_documents([unit.content for unit in units])
                await self.store.replace_document(parsed.document, units, vectors, parsed.links)
                documents_indexed += 1
                units_created += len(units)
            except Exception as exc:
                message = f"{path}: {exc}"
                errors.append(message)
                logger.exception("Failed to ingest %s", path)

        deleted = 0
        if self.config.ingestion.delete_removed_documents:
            deleted = await self.store.delete_documents_not_in(seen_ids)

        knowledge_version = stable_id(
            "|".join(sorted(fingerprints)), prefix="kv_"
        )
        return IngestionReport(
            run_id=run_id,
            documents_seen=len(files),
            documents_indexed=documents_indexed,
            units_created=units_created,
            documents_deleted=deleted,
            quarantined=quarantined,
            errors=errors,
            knowledge_version=knowledge_version,
        )
