from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from dualarrow import __version__
from dualarrow.config import load_config
from dualarrow.evaluation.runner import evaluate as run_evaluation
from dualarrow.logging import configure_logging
from dualarrow.models import QueryRequest
from dualarrow.pipeline import DualArrow

app = typer.Typer(help="DualArrow. Evidence-first adaptive RAG.", no_args_is_help=True)
config_app = typer.Typer(help="Validate and inspect configuration.")
db_app = typer.Typer(help="Database recommendations and diagnostics.")
app.add_typer(config_app, name="config")
app.add_typer(db_app, name="db")
console = Console()


def run(coro):
    return asyncio.run(coro)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"DualArrow {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show version.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "WARNING",
) -> None:
    configure_logging(log_level)


@app.command()
def init(
    directory: Annotated[Path, typer.Argument(help="Project directory.")] = Path("."),
    force: Annotated[bool, typer.Option(help="Overwrite existing sample files.")] = False,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    package_root = Path(__file__).resolve().parent
    templates = package_root / "templates" / "quickstart"
    if not templates.exists():
        raise typer.BadParameter("Quickstart templates are missing from this distribution")
    for source in templates.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(templates)
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not force:
            continue
        shutil.copy2(source, target)
    console.print(f"[green]Initialized DualArrow project in {directory.resolve()}[/green]")
    console.print("Next: dualarrow ingest --config dualarrow.yaml")


@app.command()
def ingest(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("dualarrow.yaml"),
    force: Annotated[bool, typer.Option(help="Rebuild unchanged documents.")] = False,
) -> None:
    async def execute():
        async with DualArrow.from_config(config) as engine:
            return await engine.ingest(force=force)
    report = run(execute())
    console.print_json(report.model_dump_json(indent=2))
    if report.errors:
        raise typer.Exit(code=1)


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Question to answer.")],
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("dualarrow.yaml"),
    profile: Annotated[str, typer.Option(help="Retrieval profile or auto.")] = "auto",
    evidence: Annotated[bool, typer.Option("--evidence/--no-evidence")] = True,
) -> None:
    async def execute():
        async with DualArrow.from_config(config) as engine:
            return await engine.query(
                QueryRequest(
                    query=question,
                    retrieval_profile=profile,
                    include_evidence=evidence,
                )
            )
    result = run(execute())
    console.print_json(result.model_dump_json(indent=2))


@app.command()
def serve(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("dualarrow.yaml"),
    host: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option()] = None,
    reload: Annotated[bool, typer.Option()] = False,
) -> None:
    settings = load_config(config)
    import os
    os.environ["DUALARROW_CONFIG"] = str(config.resolve())
    uvicorn.run(
        "dualarrow.api.app:create_app",
        host=host or settings.api.host,
        port=port or settings.api.port,
        reload=reload,
        workers=1 if reload else settings.api.workers,
        factory=True,
    )


@app.command()
def evaluate(
    dataset: Annotated[Path, typer.Argument(help="JSONL evaluation dataset.")],
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("dualarrow.yaml"),
    output: Annotated[Path | None, typer.Option(help="Optional JSON output file.")] = None,
    k: Annotated[int, typer.Option(help="Retrieval cutoff.")] = 10,
) -> None:
    async def execute():
        async with DualArrow.from_config(config) as engine:
            return await run_evaluation(engine, dataset, k=k)
    result = run(execute())
    serialized = json.dumps(result, indent=2, default=str)
    if output:
        output.write_text(serialized, encoding="utf-8")
        console.print(f"[green]Wrote {output}[/green]")
    else:
        console.print_json(serialized)


@app.command()
def doctor(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("dualarrow.yaml"),
) -> None:
    settings = load_config(config)
    table = Table(title="DualArrow diagnostics")
    table.add_column("Check")
    table.add_column("Result")
    table.add_row("Configuration", "OK")
    table.add_row("Knowledge root", str(settings.knowledge_store.root))
    table.add_row("Knowledge root exists", str(settings.knowledge_store.root.exists()))
    table.add_row("Sidecar", str(settings.knowledge_store.sidecar_path))
    table.add_row("Embedding provider", settings.embedding.provider)
    table.add_row("Generator", settings.generation.provider)
    table.add_row("Default profile", settings.default_profile)
    console.print(table)


@config_app.command("validate")
def config_validate(
    config: Annotated[Path, typer.Argument()] = Path("dualarrow.yaml"),
) -> None:
    settings = load_config(config)
    console.print(f"[green]Valid configuration for {settings.project.name}[/green]")


@config_app.command("explain")
def config_explain(
    config: Annotated[Path, typer.Argument()] = Path("dualarrow.yaml"),
) -> None:
    settings = load_config(config)
    console.print_json(settings.model_dump_json(indent=2))


@db_app.command("recommend")
def db_recommend(
    chunks: Annotated[int, typer.Option(help="Estimated chunk count.")] = 100_000,
    graph: Annotated[bool, typer.Option(help="Requires multi-hop graph reasoning.")] = False,
    rdf: Annotated[bool, typer.Option(help="Requires RDF/SPARQL/ontology reasoning.")] = False,
    high_qps: Annotated[bool, typer.Option(help="High query throughput.")] = False,
    minimal_ops: Annotated[bool, typer.Option(help="Prefer minimal infrastructure.")] = True,
) -> None:
    if rdf:
        stack = ["PostgreSQL", "GraphDB", "OpenSearch", "S3/MinIO"]
        profile = "semantic-rdf"
    elif graph:
        stack = ["PostgreSQL", "Qdrant", "Neo4j", "OpenSearch", "S3/MinIO"]
        profile = "graph-rag"
    elif chunks > 10_000_000 or high_qps:
        stack = ["PostgreSQL", "Milvus or Qdrant", "OpenSearch", "Redis", "S3/MinIO"]
        profile = "large-scale"
    elif minimal_ops and chunks < 2_000_000:
        stack = ["PostgreSQL + pgvector", "S3/MinIO"]
        profile = "simple-postgres"
    else:
        stack = ["PostgreSQL", "Qdrant", "OpenSearch", "Redis", "S3/MinIO"]
        profile = "balanced-production"
    console.print_json(json.dumps({"profile": profile, "stack": stack}, indent=2))


if __name__ == "__main__":
    app()
