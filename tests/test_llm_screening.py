from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from modules.alert_rules import Alert
from modules.llm_client import ChatMessage, OllamaClient
from modules.llm_screening import generate_llm_screening_alerts
from modules.patient_parser import PatientContext
from modules.vector_store import VectorSearchResult


class _FakeOllamaClient(OllamaClient):  # type: ignore[misc]
    def __init__(self, payload: str) -> None:  # pragma: no cover - trivial
        # Bypass base class network init; we only need chat() behaviour.
        self._model = "fake"
        self._host = "http://localhost"
        self._last_error: Optional[str] = None
        self._num_ctx = None
        self._payload = payload

    def chat(self, messages: List[ChatMessage]) -> Optional[str]:  # type: ignore[override]
        # Return the preconfigured payload regardless of messages.
        return self._payload


def _build_dummy_context() -> PatientContext:
    return PatientContext(
        patient_id="P005",
        name="Case 3 - AZT anemia",
        encounter_date=date(2026, 1, 22),
        art_regimen_current=["AZT", "3TC", "DTG"],
        notes_text="Heavy menses, marked pallor, Hb 7.8 g/dL on AZT.",
        latest_labs={},
    )


def _dummy_stage1_summary() -> Dict[str, Any]:
    return {
        "current_regimen": ["AZT", "3TC", "DTG"],
        "key_labs": {"hemoglobin_g_dl": 7.8},
        "key_symptoms": ["fatigue", "dizziness", "heavy menses"],
    }


def _dummy_screening_chunks() -> List[VectorSearchResult]:
    return [
        VectorSearchResult(
            chunk_id="c_101_1",
            document="Zidovudine can cause bone marrow suppression and anemia; monitor hemoglobin and consider switching if severe.",
            metadata={"page_number": 101},
            distance=0.1,
        )
    ]


def test_llm_screening_maps_issues_to_alerts() -> None:
    payload = (
        "[{"  # start JSON array
        "\"alert_id\": \"llm_azt_anemia\","
        "\"title\": \"Severe anemia on AZT\","
        "\"message\": \"Hb 7.8 g/dL with heavy menses in a woman on AZT suggests clinically significant anemia requiring review.\","
        "\"issue_type\": \"toxicity\","
        "\"severity\": \"high\","
        "\"recommended_action\": \"Evaluate for AZT-associated anemia, manage bleeding, and consider switching off AZT.\","
        "\"citations\": [{\"page_number\": 101, \"chunk_id\": \"c_101_1\"}]"
        "}]"
    )
    client = _FakeOllamaClient(payload)
    ctx = _build_dummy_context()

    alerts = generate_llm_screening_alerts(
        patient_context=ctx,
        stage1_summary=_dummy_stage1_summary(),
        screening_chunks=_dummy_screening_chunks(),
        llm_client=client,
    )

    assert alerts, "Expected at least one LLM screening alert"
    a: Alert = alerts[0]
    assert a.alert_id == "llm_azt_anemia"
    assert "AZT" in a.title or "azt" in a.title.lower()
    assert a.evidence.get("type") == "llm_screening"
    assert a.evidence.get("severity") == "high"
    assert a.evidence.get("citations"), "Expected citations from guideline chunks"


def test_llm_screening_empty_json_returns_no_alerts() -> None:
    client = _FakeOllamaClient("[]")
    ctx = _build_dummy_context()

    alerts = generate_llm_screening_alerts(
        patient_context=ctx,
        stage1_summary=_dummy_stage1_summary(),
        screening_chunks=_dummy_screening_chunks(),
        llm_client=client,
    )

    assert alerts == []
