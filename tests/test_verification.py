from dualarrow.config import VerificationConfig
from dualarrow.models import Claim, DraftAnswer, EvidenceCandidate
from dualarrow.verification.verifier import EvidenceVerifier


async def test_number_mismatch_is_not_fully_supported():
    verifier = EvidenceVerifier(
        VerificationConfig(minimum_support_score=0.3, minimum_confidence=0.3)
    )
    evidence = [
        EvidenceCandidate(
            evidence_id="ev_1",
            unit_id="u1",
            document_id="d1",
            content="Stop the machine when vibration exceeds 8.0 mm/s RMS.",
            source_uri="manual.md",
            content_hash="hash",
            authority_score=1,
        )
    ]
    draft = DraftAnswer(
        answer="Stop at 10.0 mm/s RMS.",
        claims=[Claim(claim_id="c1", text="Stop at 10.0 mm/s RMS.", evidence_ids=["ev_1"])],
    )
    result = await verifier.verify("When should it stop?", draft, evidence)
    assert result.claims[0].support_score < 0.6
    assert any("numbers_not_found" in note for note in result.claims[0].notes)
