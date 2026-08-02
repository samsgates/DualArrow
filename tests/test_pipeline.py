from dualarrow.models import AnswerStatus, QueryRequest
from dualarrow.pipeline import DualArrow


async def test_ingest_and_query(project):
    async with DualArrow.from_config(project / "dualarrow.yaml") as engine:
        report = await engine.ingest()
        assert report.documents_indexed == 1
        assert report.units_created > 0

        result = await engine.query(
            QueryRequest(query="Which bearing is approved for AX-100?")
        )
        assert result.status in {
            AnswerStatus.VERIFIED_ANSWER,
            AnswerStatus.VERIFIED_PARTIAL_ANSWER,
        }
        assert "BR-900" in result.answer
        assert result.citations
        assert result.trace_id.startswith("trace_")


async def test_incremental_ingestion(project):
    async with DualArrow.from_config(project / "dualarrow.yaml") as engine:
        first = await engine.ingest()
        second = await engine.ingest()
        assert first.documents_indexed == 1
        assert second.documents_indexed == 0
