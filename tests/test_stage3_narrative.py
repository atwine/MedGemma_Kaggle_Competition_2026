from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List, Optional

from modules.patient_parser import build_patient_context
from modules.stage1_summary import build_stage1_summary
from modules.stage3_narrative import generate_stage3_narrative
from modules.vector_store import VectorSearchResult


class _StubLLM:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload
        self.last_error: Optional[str] = None

    def chat(self, messages: List[Dict[str, Any]]) -> str:
        return json.dumps(self._payload)


def _make_patient() -> Dict[str, Any]:
    return {
        "patient_id": "p1",
        "name": "Test Patient",
        "today_encounter": {"date": "2024-01-01", "note": "Plan: recheck creatinine."},
        "visits": [
            {"date": "2023-10-01", "clinician_note": "Sleep disturbance. Plan: counsel."}
        ],
        "labs": {
            "creatinine": [{"date": "2023-01-01", "value_umol_per_l": 82}],
            "viral_load": [{"date": "2023-11-01", "value_copies_per_ml": 200}],
        },
        "art_regimen_current": ["TDF", "3TC", "EFV"],
    }


def test_generate_stage3_narrative_keys_and_issue_fields() -> None:
    patient = _make_patient()
    ctx = build_patient_context(patient)
    stage1 = build_stage1_summary(patient=patient, context=ctx)

    evidence_map = {
        "tdf_creatinine_overdue": [
            VectorSearchResult(
                chunk_id="c105",
                document="Guideline: check creatinine at least annually for patients on TDF.",
                metadata={"page_number": 105},
                distance=0.2,
            )
        ]
    }

    payload = {
        "Subjective": ["Sleep disturbance"],
        "Objective": ["VL=200 copies/mL", "Creatinine last 2023-01-01"],
        "Assessment": [
            {
                "issue_type": "monitoring_gap",
                "severity": "medium",
                "guideline_reference": "pp. 99–114",
                "already_in_plan": "no",
                "nudge_needed": "yes",
            }
        ],
        "Plan": ["Order creatinine"]
    }

    llm = _StubLLM(payload)

    out = generate_stage3_narrative(
        patient_context=ctx,
        stage1_summary=stage1,
        evidence_map=evidence_map,
        llm_client=llm,  # type: ignore[arg-type]
    )

    assert isinstance(out, dict)
    for key in ("Subjective", "Objective", "Assessment", "Plan"):
        assert key in out
    assert isinstance(out["Assessment"], list) and len(out["Assessment"]) >= 1
    issue = out["Assessment"][0]
    for k in ("issue_type", "severity", "guideline_reference", "already_in_plan", "nudge_needed"):
        assert k in issue


def test_generate_stage3_narrative_returns_none_without_llm() -> None:
    patient = _make_patient()
    ctx = build_patient_context(patient)
    stage1 = build_stage1_summary(patient=patient, context=ctx)

    evidence_map: Dict[str, List[VectorSearchResult]] = {}

    out = generate_stage3_narrative(
        patient_context=ctx,
        stage1_summary=stage1,
        evidence_map=evidence_map,
        llm_client=None,
    )

    assert out is None


def test_generate_stage3_narrative_parses_fenced_json() -> None:
    patient = _make_patient()
    ctx = build_patient_context(patient)
    stage1 = build_stage1_summary(patient=patient, context=ctx)

    evidence_map: Dict[str, List[VectorSearchResult]] = {}

    class _FencedLLM:
        last_error: Optional[str] = None

        def chat(self, messages: List[Dict[str, Any]]) -> str:
            body = {
                "Subjective": ["No complaints"],
                "Objective": ["Stable"],
                "Assessment": [],
                "Plan": ["Follow up"]
            }
            return "```json\n" + json.dumps(body) + "\n```"

    out = generate_stage3_narrative(
        patient_context=ctx,
        stage1_summary=stage1,
        evidence_map=evidence_map,
        llm_client=_FencedLLM(),  # type: ignore[arg-type]
    )

    assert isinstance(out, dict)
    for key in ("Subjective", "Objective", "Assessment", "Plan"):
        assert key in out
