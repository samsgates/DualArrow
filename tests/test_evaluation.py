from pathlib import Path

from dualarrow.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank, token_f1
from dualarrow.evaluation.runner import evaluate
from dualarrow.pipeline import DualArrow


def test_metrics():
    retrieved = ["a", "b", "c"]
    relevant = {"b", "c"}
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert 0 < ndcg_at_k(retrieved, relevant, 3) <= 1
    assert token_f1("approved bearing BR-900", "bearing BR-900") > 0.7


async def test_evaluation_runner(project: Path):
    dataset = project / "dataset.jsonl"
    dataset.write_text(
        '{"query":"Which bearing is approved for AX-100?","expected_answer":"The approved bearing is BR-900."}\n',
        encoding="utf-8",
    )
    async with DualArrow.from_config(project / "dualarrow.yaml") as engine:
        await engine.ingest()
        result = await evaluate(engine, dataset)
    assert result["count"] == 1
    assert result["summary"]["answer_f1"] > 0
