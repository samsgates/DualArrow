from __future__ import annotations

from contextlib import closing
import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from dualarrow.models import EvidenceCandidate, KnowledgeUnit, SourceDocument
from dualarrow.stores.base import StoreCapability
from dualarrow.utils import json_dumps, stable_id, tokenize


class ArrowMDStore:
    capabilities = {
        StoreCapability.DOCUMENTS,
        StoreCapability.DENSE_VECTOR,
        StoreCapability.BM25,
        StoreCapability.FULL_TEXT,
        StoreCapability.METADATA_FILTER,
        StoreCapability.GRAPH_TRAVERSAL,
        StoreCapability.TRANSACTIONS,
        StoreCapability.ACCESS_CONTROL,
    }

    def __init__(self, path: Path, dimensions: int, *, ephemeral: bool = False) -> None:
        self.path = path
        self.dimensions = dimensions
        self.ephemeral = ephemeral
        self._fts_available = True

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        with closing(self._connect()) as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    version TEXT NOT NULL,
                    authority REAL NOT NULL,
                    status TEXT NOT NULL,
                    effective_from TEXT,
                    expires_at TEXT,
                    metadata_json TEXT NOT NULL,
                    modified_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS units (
                    unit_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                    view_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    title TEXT NOT NULL,
                    parent_unit_id TEXT,
                    heading_path_json TEXT NOT NULL,
                    line_start INTEGER,
                    line_end INTEGER,
                    authority REAL NOT NULL,
                    effective_from TEXT,
                    expires_at TEXT,
                    metadata_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_units_document ON units(document_id);
                CREATE INDEX IF NOT EXISTS idx_units_view ON units(view_id);
                CREATE TABLE IF NOT EXISTS embeddings (
                    unit_id TEXT PRIMARY KEY REFERENCES units(unit_id) ON DELETE CASCADE,
                    dimensions INTEGER NOT NULL,
                    vector BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS links (
                    source_document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                    target TEXT NOT NULL,
                    PRIMARY KEY(source_document_id, target)
                );
                CREATE TABLE IF NOT EXISTS manifest (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            try:
                db.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS units_fts USING fts5("
                    "unit_id UNINDEXED, title, heading_path, content, tokenize='unicode61')"
                )
            except sqlite3.OperationalError:
                self._fts_available = False

    async def close(self) -> None:
        if not self.ephemeral:
            return
        for candidate in (self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm")):
            candidate.unlink(missing_ok=True)

    async def replace_document(
        self,
        document: SourceDocument,
        units: list[KnowledgeUnit],
        embeddings: np.ndarray,
        links: list[str],
    ) -> None:
        if len(units) != len(embeddings):
            raise ValueError("units and embeddings must have the same length")
        await asyncio.to_thread(self._replace_document_sync, document, units, embeddings, links)

    def _replace_document_sync(
        self,
        document: SourceDocument,
        units: list[KnowledgeUnit],
        embeddings: np.ndarray,
        links: list[str],
    ) -> None:
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            old_units = [
                row[0] for row in db.execute(
                    "SELECT unit_id FROM units WHERE document_id = ?", (document.document_id,)
                )
            ]
            if self._fts_available and old_units:
                db.executemany("DELETE FROM units_fts WHERE unit_id = ?", [(item,) for item in old_units])
            db.execute("DELETE FROM documents WHERE document_id = ?", (document.document_id,))
            db.execute(
                """
                INSERT INTO documents (
                    document_id,title,content,source_uri,checksum,version,authority,status,
                    effective_from,expires_at,metadata_json,modified_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    document.document_id,
                    document.title,
                    document.content,
                    document.source_uri,
                    document.checksum,
                    document.version,
                    document.authority,
                    document.status,
                    document.effective_from.isoformat() if document.effective_from else None,
                    document.expires_at.isoformat() if document.expires_at else None,
                    json_dumps(document.metadata),
                    document.modified_at.isoformat(),
                ),
            )
            for unit, vector in zip(units, embeddings, strict=True):
                db.execute(
                    """
                    INSERT INTO units (
                        unit_id,document_id,view_id,content,source_uri,checksum,title,parent_unit_id,
                        heading_path_json,line_start,line_end,authority,effective_from,expires_at,metadata_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        unit.unit_id,
                        unit.document_id,
                        unit.view_id,
                        unit.content,
                        unit.source_uri,
                        unit.checksum,
                        unit.title,
                        unit.parent_unit_id,
                        json_dumps(unit.heading_path),
                        unit.line_start,
                        unit.line_end,
                        unit.authority,
                        unit.effective_from.isoformat() if unit.effective_from else None,
                        unit.expires_at.isoformat() if unit.expires_at else None,
                        json_dumps(unit.metadata),
                    ),
                )
                vector32 = np.asarray(vector, dtype=np.float32)
                db.execute(
                    "INSERT INTO embeddings(unit_id,dimensions,vector) VALUES (?,?,?)",
                    (unit.unit_id, len(vector32), vector32.tobytes()),
                )
                if self._fts_available:
                    db.execute(
                        "INSERT INTO units_fts(unit_id,title,heading_path,content) VALUES (?,?,?,?)",
                        (unit.unit_id, unit.title, " / ".join(unit.heading_path), unit.content),
                    )
            db.executemany(
                "INSERT OR IGNORE INTO links(source_document_id,target) VALUES (?,?)",
                [(document.document_id, target) for target in links],
            )
            db.execute(
                "INSERT OR REPLACE INTO manifest(key,value) VALUES ('updated_at',?)",
                (datetime.now(UTC).isoformat(),),
            )
            db.commit()

    async def document_checksum(self, document_id: str) -> str | None:
        return await asyncio.to_thread(self._document_checksum_sync, document_id)

    def _document_checksum_sync(self, document_id: str) -> str | None:
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT checksum FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
            return str(row[0]) if row else None

    async def list_document_ids(self) -> set[str]:
        return await asyncio.to_thread(self._list_document_ids_sync)

    def _list_document_ids_sync(self) -> set[str]:
        with closing(self._connect()) as db:
            return {str(row[0]) for row in db.execute("SELECT document_id FROM documents")}

    async def delete_documents_not_in(self, document_ids: set[str]) -> int:
        return await asyncio.to_thread(self._delete_documents_not_in_sync, document_ids)

    def _delete_documents_not_in_sync(self, document_ids: set[str]) -> int:
        with closing(self._connect()) as db:
            existing = {str(row[0]) for row in db.execute("SELECT document_id FROM documents")}
            deleted = existing - document_ids
            for document_id in deleted:
                unit_ids = [
                    row[0] for row in db.execute(
                        "SELECT unit_id FROM units WHERE document_id = ?", (document_id,)
                    )
                ]
                if self._fts_available:
                    db.executemany("DELETE FROM units_fts WHERE unit_id = ?", [(u,) for u in unit_ids])
                db.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            db.commit()
            return len(deleted)

    async def exact_search(self, query: str, top_k: int, filters: dict) -> list[EvidenceCandidate]:
        return await asyncio.to_thread(self._exact_search_sync, query, top_k, filters)

    def _exact_search_sync(self, query: str, top_k: int, filters: dict) -> list[EvidenceCandidate]:
        pattern = f"%{query.casefold()}%"
        with closing(self._connect()) as db:
            rows = db.execute(
                """
                SELECT u.*, CASE
                    WHEN lower(u.title) = ? THEN 5.0
                    WHEN lower(u.content) LIKE ? THEN 2.0
                    ELSE 1.0 END AS score
                FROM units u JOIN documents d ON d.document_id=u.document_id
                WHERE (lower(u.content) LIKE ? OR lower(u.title) LIKE ?)
                  AND d.status='approved'
                ORDER BY score DESC, length(u.content) ASC
                LIMIT ?
                """,
                (query.casefold(), pattern, pattern, pattern, top_k),
            ).fetchall()
        return [self._row_to_evidence(row, "exact", float(row["score"])) for row in rows]

    async def lexical_search(self, query: str, top_k: int, filters: dict) -> list[EvidenceCandidate]:
        return await asyncio.to_thread(self._lexical_search_sync, query, top_k, filters)

    def _lexical_search_sync(self, query: str, top_k: int, filters: dict) -> list[EvidenceCandidate]:
        tokens = [token for token in tokenize(query) if len(token) > 1]
        if not tokens:
            return []
        with closing(self._connect()) as db:
            if self._fts_available:
                fts_query = " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens[:32])
                try:
                    rows = db.execute(
                        """
                        SELECT u.*, -bm25(units_fts, 0.0, 2.5, 1.5, 1.0) AS score
                        FROM units_fts JOIN units u ON u.unit_id=units_fts.unit_id
                        JOIN documents d ON d.document_id=u.document_id
                        WHERE units_fts MATCH ? AND d.status='approved'
                        ORDER BY score DESC LIMIT ?
                        """,
                        (fts_query, top_k),
                    ).fetchall()
                    return [self._row_to_evidence(row, "bm25", float(row["score"])) for row in rows]
                except sqlite3.OperationalError:
                    pass
            return self._fallback_bm25(db, tokens, top_k)

    def _fallback_bm25(
        self, db: sqlite3.Connection, tokens: list[str], top_k: int
    ) -> list[EvidenceCandidate]:
        rows = db.execute(
            "SELECT u.* FROM units u JOIN documents d ON d.document_id=u.document_id "
            "WHERE d.status='approved'"
        ).fetchall()
        scored: list[tuple[float, sqlite3.Row]] = []
        query_set = set(tokens)
        for row in rows:
            words = tokenize(str(row["title"]) + " " + str(row["content"]))
            if not words:
                continue
            overlap = sum(1 for token in words if token in query_set)
            if overlap:
                scored.append((overlap / (len(words) ** 0.35), row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._row_to_evidence(row, "bm25", score) for score, row in scored[:top_k]]

    async def dense_search(self, vector: np.ndarray, top_k: int, filters: dict) -> list[EvidenceCandidate]:
        return await asyncio.to_thread(self._dense_search_sync, vector, top_k, filters)

    def _dense_search_sync(self, vector: np.ndarray, top_k: int, filters: dict) -> list[EvidenceCandidate]:
        query = np.asarray(vector, dtype=np.float32)
        query_norm = float(np.linalg.norm(query))
        if not query_norm:
            return []
        with closing(self._connect()) as db:
            rows = db.execute(
                """
                SELECT u.*, e.dimensions, e.vector
                FROM units u JOIN embeddings e ON e.unit_id=u.unit_id
                JOIN documents d ON d.document_id=u.document_id
                WHERE d.status='approved'
                """
            ).fetchall()
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            candidate = np.frombuffer(row["vector"], dtype=np.float32)
            if candidate.shape != query.shape:
                continue
            denominator = float(np.linalg.norm(candidate)) * query_norm
            score = float(np.dot(candidate, query) / denominator) if denominator else 0.0
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._row_to_evidence(row, "dense", score) for score, row in scored[:top_k]]

    async def graph_search(self, query: str, top_k: int, filters: dict) -> list[EvidenceCandidate]:
        return await asyncio.to_thread(self._graph_search_sync, query, top_k, filters)

    def _graph_search_sync(self, query: str, top_k: int, filters: dict) -> list[EvidenceCandidate]:
        tokens = set(tokenize(query))
        if not tokens:
            return []
        with closing(self._connect()) as db:
            links = db.execute("SELECT source_document_id,target FROM links").fetchall()
            document_rows = db.execute("SELECT document_id,title,source_uri,metadata_json FROM documents").fetchall()
            doc_lookup = {str(row["document_id"]): row for row in document_rows}
            path_lookup: dict[str, str] = {}
            for row in document_rows:
                metadata = json.loads(row["metadata_json"])
                relative = metadata.get("relative_path")
                if relative:
                    path_lookup[str(relative)] = str(row["document_id"])
                    path_lookup[Path(str(relative)).stem] = str(row["document_id"])
            scores: dict[str, float] = {}
            for row in document_rows:
                haystack = set(tokenize(str(row["title"]) + " " + str(row["metadata_json"])))
                overlap = len(tokens & haystack)
                if overlap:
                    scores[str(row["document_id"])] = float(overlap)
            seeds = set(scores)
            for link in links:
                source = str(link["source_document_id"])
                target_raw = str(link["target"])
                target = path_lookup.get(target_raw) or path_lookup.get(Path(target_raw).stem)
                if source in seeds and target:
                    scores[target] = scores.get(target, 0.0) + 0.75
                if target in seeds:
                    scores[source] = scores.get(source, 0.0) + 0.5
            if not scores:
                return []
            ranked_docs = sorted(scores, key=scores.get, reverse=True)[: max(1, top_k // 2)]
            placeholders = ",".join("?" for _ in ranked_docs)
            unit_rows = db.execute(
                f"SELECT * FROM units WHERE document_id IN ({placeholders}) LIMIT ?",
                (*ranked_docs, top_k),
            ).fetchall()
        return [
            self._row_to_evidence(row, "graph", scores.get(str(row["document_id"]), 0.0))
            for row in unit_rows
        ]

    async def get_units(self, unit_ids: list[str]) -> list[KnowledgeUnit]:
        if not unit_ids:
            return []
        return await asyncio.to_thread(self._get_units_sync, unit_ids)

    def _get_units_sync(self, unit_ids: list[str]) -> list[KnowledgeUnit]:
        with closing(self._connect()) as db:
            placeholders = ",".join("?" for _ in unit_ids)
            rows = db.execute(
                f"SELECT * FROM units WHERE unit_id IN ({placeholders})", tuple(unit_ids)
            ).fetchall()
        by_id = {str(row["unit_id"]): self._row_to_unit(row) for row in rows}
        return [by_id[item] for item in unit_ids if item in by_id]

    def _row_to_unit(self, row: sqlite3.Row) -> KnowledgeUnit:
        return KnowledgeUnit(
            unit_id=str(row["unit_id"]),
            document_id=str(row["document_id"]),
            view_id=str(row["view_id"]),
            content=str(row["content"]),
            source_uri=str(row["source_uri"]),
            checksum=str(row["checksum"]),
            title=str(row["title"]),
            parent_unit_id=row["parent_unit_id"],
            heading_path=json.loads(row["heading_path_json"]),
            line_start=row["line_start"],
            line_end=row["line_end"],
            authority=float(row["authority"]),
            effective_from=row["effective_from"],
            expires_at=row["expires_at"],
            metadata=json.loads(row["metadata_json"]),
        )

    def _row_to_evidence(self, row: sqlite3.Row, method: str, score: float) -> EvidenceCandidate:
        return EvidenceCandidate(
            evidence_id=stable_id(str(row["unit_id"]), prefix="ev_"),
            unit_id=str(row["unit_id"]),
            document_id=str(row["document_id"]),
            content=str(row["content"]),
            source_uri=str(row["source_uri"]),
            title=str(row["title"]),
            heading_path=json.loads(row["heading_path_json"]),
            line_start=row["line_start"],
            line_end=row["line_end"],
            retrieval_methods=[method],
            scores={method: score},
            normalized_score=score,
            authority_score=float(row["authority"]),
            freshness_score=self._freshness(row["effective_from"], row["expires_at"]),
            provenance_score=1.0,
            content_hash=str(row["checksum"]),
            metadata=json.loads(row["metadata_json"]),
        )

    @staticmethod
    def _freshness(effective_from: str | None, expires_at: str | None) -> float:
        now = datetime.now(UTC)
        if expires_at:
            expiry = datetime.fromisoformat(expires_at)
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=UTC)
            if expiry < now:
                return 0.1
        if effective_from:
            effective = datetime.fromisoformat(effective_from)
            if effective.tzinfo is None:
                effective = effective.replace(tzinfo=UTC)
            if effective > now:
                return 0.2
        return 1.0
