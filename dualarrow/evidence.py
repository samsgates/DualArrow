from __future__ import annotations

from contextlib import closing
import asyncio
import sqlite3
from pathlib import Path

from dualarrow.models import AnswerResult, RetrievalTrace


class EvidenceLedger:
    def __init__(self, path: Path, *, ephemeral: bool = False) -> None:
        self.path = path
        self.ephemeral = ephemeral

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        with closing(self._connect()) as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS query_traces (
                    trace_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    selected_profile TEXT NOT NULL,
                    trace_json TEXT NOT NULL,
                    answer_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )


    async def close(self) -> None:
        if not self.ephemeral:
            return
        for candidate in (self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm")):
            candidate.unlink(missing_ok=True)

    async def record(self, trace: RetrievalTrace, answer: AnswerResult) -> None:
        await asyncio.to_thread(self._record_sync, trace, answer)

    def _record_sync(self, trace: RetrievalTrace, answer: AnswerResult) -> None:
        with closing(self._connect()) as db:
            db.execute(
                "INSERT OR REPLACE INTO query_traces(trace_id,query,selected_profile,trace_json,answer_json) "
                "VALUES (?,?,?,?,?)",
                (
                    trace.trace_id,
                    trace.query,
                    trace.selected_profile,
                    trace.model_dump_json(),
                    answer.model_dump_json(),
                ),
            )
