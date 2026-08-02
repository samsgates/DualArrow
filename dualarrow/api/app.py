from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from dualarrow import __version__
from dualarrow.errors import DualArrowError
from dualarrow.models import AnswerResult, IngestionReport, QueryRequest
from dualarrow.pipeline import DualArrow


def create_app(config_path: str | Path | None = None) -> FastAPI:
    resolved = Path(config_path or os.getenv("DUALARROW_CONFIG", "dualarrow.yaml"))
    engine = DualArrow.from_config(resolved)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await engine.start()
        app.state.engine = engine
        yield
        await engine.close()

    app = FastAPI(
        title="DualArrow API",
        version=__version__,
        description="Evidence-first adaptive RAG API",
        lifespan=lifespan,
    )

    async def authorize(x_api_key: str | None = Header(default=None)) -> None:
        expected = engine.config.security.api_key
        if expected and (x_api_key is None or not secrets.compare_digest(x_api_key, expected)):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    @app.exception_handler(DualArrowError)
    async def framework_error(_: Request, exc: DualArrowError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": type(exc).__name__, "detail": str(exc)})

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "project": engine.config.project.name,
            "started": engine._started,
        }

    @app.post("/v1/ingest", response_model=IngestionReport, dependencies=[Depends(authorize)])
    async def ingest(force: bool = False) -> IngestionReport:
        return await engine.ingest(force=force)

    @app.post("/v1/query", response_model=AnswerResult, dependencies=[Depends(authorize)])
    async def query(request: QueryRequest) -> AnswerResult:
        return await engine.query(request)

    @app.get("/v1/config", dependencies=[Depends(authorize)])
    async def config() -> dict:
        value = engine.config.model_dump(mode="json")
        if value.get("security"):
            value["security"]["api_key"] = "***" if value["security"].get("api_key") else None
        if value.get("embedding"):
            value["embedding"]["api_key"] = "***" if value["embedding"].get("api_key") else None
        if value.get("generation"):
            value["generation"]["api_key"] = "***" if value["generation"].get("api_key") else None
        return value

    return app

