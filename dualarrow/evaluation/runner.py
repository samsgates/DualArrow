from __future__ import annotations

from pathlib import Path
from statistics import mean

from pydantic import Field

from dualarrow.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank, token_f1
from dualarrow.models import QueryRequest, StrictModel
from dualarrow.pipeline import DualArrow


class EvaluationItem(StrictModel):
    query: str
    relevant_unit_ids: list[str] = Field(default_factory=list)
    expected_answer: str | None = None
    profile: str = "auto"


async def evaluate(engine: DualArrow, dataset: Path, k: int = 10) -> dict:
    items = [
        EvaluationItem.model_validate_json(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    for item in items:
        request = QueryRequest(
            query=item.query,
            retrieval_profile=item.profile,
            include_evidence=True,
        )
        answer = await engine.query(request)
        retrieved = [candidate.unit_id for candidate in answer.evidence]
        relevant = set(item.relevant_unit_ids)
        row = {
            "query": item.query,
            "status": answer.status,
            "precision_at_k": precision_at_k(retrieved, relevant, k) if relevant else None,
            "recall_at_k": recall_at_k(retrieved, relevant, k) if relevant else None,
            "mrr": reciprocal_rank(retrieved, relevant) if relevant else None,
            "ndcg_at_k": ndcg_at_k(retrieved, relevant, k) if relevant else None,
            "answer_f1": token_f1(answer.answer, item.expected_answer) if item.expected_answer else None,
            "confidence": answer.confidence.overall,
        }
        rows.append(row)

    def average(key: str) -> float | None:
        values = [float(row[key]) for row in rows if row[key] is not None]
        return mean(values) if values else None

    return {
        "count": len(rows),
        "summary": {
            "precision_at_k": average("precision_at_k"),
            "recall_at_k": average("recall_at_k"),
            "mrr": average("mrr"),
            "ndcg_at_k": average("ndcg_at_k"),
            "answer_f1": average("answer_f1"),
            "confidence": average("confidence"),
        },
        "items": rows,
    }
