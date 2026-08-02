from __future__ import annotations

import math

from dualarrow.utils import tokenize


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    items = retrieved[:k]
    return sum(1 for item in items if item in relevant) / max(1, len(items))


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return sum(1 for item in retrieved[:k] if item in relevant) / max(1, len(relevant))


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for index, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(index + 2)
        for index, item in enumerate(retrieved[:k])
        if item in relevant
    )
    ideal_count = min(k, len(relevant))
    ideal = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
    return dcg / ideal if ideal else 0.0


def token_f1(predicted: str, expected: str) -> float:
    predicted_tokens = tokenize(predicted)
    expected_tokens = tokenize(expected)
    if not predicted_tokens or not expected_tokens:
        return float(predicted_tokens == expected_tokens)
    predicted_counts = {token: predicted_tokens.count(token) for token in set(predicted_tokens)}
    expected_counts = {token: expected_tokens.count(token) for token in set(expected_tokens)}
    common = sum(min(count, expected_counts.get(token, 0)) for token, count in predicted_counts.items())
    precision = common / len(predicted_tokens)
    recall = common / len(expected_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
