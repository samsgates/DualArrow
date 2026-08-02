from __future__ import annotations

import json
import re

import httpx

from dualarrow.models import Claim, DraftAnswer, EvidenceCandidate
from dualarrow.utils import stable_id

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class OpenAICompatibleGenerator:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None,
        temperature: float = 0.0,
        timeout: float = 60,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def generate(self, query: str, evidence: list[EvidenceCandidate]) -> DraftAnswer:
        context = "\n\n".join(
            f"EVIDENCE_ID: {item.evidence_id}\nSOURCE: {item.source_uri}\nCONTENT:\n{item.content}"
            for item in evidence
        )
        system = (
            "You are DualArrow's evidence-constrained answer generator. Use only supplied evidence. "
            "Do not use model memory. Preserve qualifications, negation, dates, numbers, and units. "
            "Return JSON with keys answer and claims. Each claim must contain text and evidence_ids. "
            "Never cite an evidence ID that is not supplied. If evidence is insufficient, return an empty "
            "answer and empty claims."
        )
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"QUESTION:\n{query}\n\n{context}"},
            ],
        }
        response = await self.client.post("/chat/completions", json=payload)
        if response.status_code == 400:
            payload.pop("response_format", None)
            response = await self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        match = _JSON_RE.search(text)
        if not match:
            raise ValueError("Generator did not return JSON")
        data = json.loads(match.group(0))
        valid_ids = {item.evidence_id for item in evidence}
        claims = []
        for raw in data.get("claims", []):
            claim_text = str(raw.get("text", "")).strip()
            evidence_ids = [str(item) for item in raw.get("evidence_ids", []) if str(item) in valid_ids]
            if claim_text:
                claims.append(
                    Claim(
                        claim_id=stable_id(claim_text, prefix="claim_"),
                        text=claim_text,
                        evidence_ids=evidence_ids,
                    )
                )
        return DraftAnswer(answer=str(data.get("answer", "")).strip(), claims=claims)
