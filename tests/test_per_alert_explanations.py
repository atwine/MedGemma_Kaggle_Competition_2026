from __future__ import annotations

from datetime import date
from typing import List, Dict, Any, Optional

from modules.alert_rules import Alert
from modules.patient_parser import PatientContext
from modules.vector_store import VectorSearchResult
from modules.explanation_generator import generate_explanation


class _StubLLM:
    """Minimal stub for OllamaClient used in tests.
    Provides a .chat() method and .last_error attribute.
    """

    def __init__(self, reply_text: str) -> None:
        self._reply = reply_text
        self.last_error: Optional[str] = None

    def chat(self, messages: List[Dict[str, Any]]) -> str:
        return self._reply


def _make_patient_context() -> PatientContext:
    return PatientContext(
        patient_id="p1",
        name="Test Patient",
        encounter_date=date(2024, 1, 1),
        art_regimen_current=["TDF"],
        notes_text="Some clinical notes...",
        latest_labs={},
    )


def test_generate_explanation_appends_citations_when_llm_omits_and_chunks_exist() -> None:
    # Arrange: retrieved chunk with page and text; LLM reply without citations section
    ctx = _make_patient_context()
    alert = Alert(
        alert_id="a1",
        title="Creatinine monitoring overdue while on TDF",
        message="Creatinine last recorded 500 days ago for a patient on TDF.",
        evidence={"regimen": ["TDF"], "days_since_creatinine": 500},
        query_hint="TDF renal function monitoring creatinine schedule",
    )
    retrieved = [
        VectorSearchResult(
            chunk_id="c1",
            document="Guideline: check creatinine at least annually for patients on TDF.",
            metadata={"page_number": 105},
            distance=0.12,
        )
    ]
    llm = _StubLLM("Short explanation without explicit citations.")

    # Act
    result = generate_explanation(
        patient_context=ctx,
        alert=alert,
        retrieved_chunks=retrieved,
        llm_client=llm,  # type: ignore[arg-type]
    )

    # Assert: used LLM and appended a deterministic Citations block
    assert result.used_llm is True
    assert "Citations:" in result.text
    assert "chunk_id=c1" in result.text
    assert "page=105" in result.text


def test_generate_explanation_falls_back_when_no_chunks() -> None:
    # Arrange: no retrieved chunks -> do not call LLM
    ctx = _make_patient_context()
    alert = Alert(
        alert_id="a2",
        title="EFV side effects",
        message="Sleep disturbance mentioned in notes for a patient on EFV.",
        evidence={"regimen": ["EFV"], "keyword_hits": ["sleep"]},
        query_hint="Efavirenz sleep disturbance vivid dreams management",
    )

    # Act
    result = generate_explanation(
        patient_context=ctx,
        alert=alert,
        retrieved_chunks=[],
        llm_client=_StubLLM("anything"),  # should be ignored
    )

    # Assert: deterministic fallback
    assert result.used_llm is False
    assert "Guideline excerpts: none retrieved" in result.text
