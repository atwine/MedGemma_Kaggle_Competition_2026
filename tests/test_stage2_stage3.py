from __future__ import annotations

import json
from datetime import date
from typing import List, Dict, Any, Optional

from modules.guideline_processor import GuidelineChunk
from modules.vector_store import InMemoryVectorStore, VectorSearchResult
from modules.explanation_generator import generate_stage3_synthesis_issues
from modules.alert_rules import Alert
from modules.patient_parser import PatientContext


class _DummyEmbedder:
    """Deterministic, lightweight embedder for tests (no external models)."""

    def encode(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            s = t or ""
            out.append([float(len(s)), float(s.count("a")), float(s.count("e"))])
        return out


def _make_patient_context() -> PatientContext:
    return PatientContext(
        patient_id="p1",
        name="Test Patient",
        encounter_date=date(2024, 1, 1),
        art_regimen_current=["TDF"],
        notes_text="Some clinical notes...",
        latest_labs={},
    )


def test_page_range_filter_bounds_in_memory_store() -> None:
    store = InMemoryVectorStore()
    embedder = _DummyEmbedder()

    chunks = [
        GuidelineChunk(chunk_id="c50", text="x" * 10, source_path="/tmp/g.pdf", page_number=50),
        GuidelineChunk(chunk_id="c99", text="alpha guidance", source_path="/tmp/g.pdf", page_number=99),
        GuidelineChunk(chunk_id="c105", text="beta guidance", source_path="/tmp/g.pdf", page_number=105),
        GuidelineChunk(chunk_id="c114", text="gamma guidance", source_path="/tmp/g.pdf", page_number=114),
        GuidelineChunk(chunk_id="c150", text="delta guidance", source_path="/tmp/g.pdf", page_number=150),
    ]
    store.index_guidelines(chunks, embedder)  # type: ignore[arg-type]

    results = store.query("guidance", embedder, top_k=10, page_range=(99, 114))  # type: ignore[arg-type]
    assert results
    assert all(int(r.metadata.get("page_number")) >= 99 and int(r.metadata.get("page_number")) <= 114 for r in results)

    # Out-of-range yields empty results
    results_empty = store.query("guidance", embedder, top_k=10, page_range=(200, 210))  # type: ignore[arg-type]
    assert results_empty == []


class _StubLLM:
    def __init__(self, payload: List[Dict[str, Any]]) -> None:
        self._payload = payload
        self.last_error: Optional[str] = None

    def chat(self, messages: List[Dict[str, Any]]) -> str:
        return json.dumps(self._payload)


def test_synthesis_returns_valid_issues_mapped_to_alerts() -> None:
    ctx = _make_patient_context()

    # Deterministic alerts (Stage 1 summary)
    deterministic_alerts = [
        Alert(
            alert_id="tdf_creatinine_overdue",
            title="Creatinine monitoring overdue while on TDF",
            message="Creatinine last recorded 400 days ago.",
            evidence={"days_since_creatinine": 400},
            query_hint="TDF renal function monitoring creatinine schedule",
        )
    ]

    # Evidence map (Stage 2 compact bundle)
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

    # LLM payload (Stage 3 issues JSON)
    payload: List[Dict[str, Any]] = [
        {
            "issue_type": "monitoring_gap",
            "severity": "medium",
            "guideline_reference": "pp. 99–114",
            "already_in_plan": "no",
            "nudge_needed": "yes",
        }
    ]

    llm = _StubLLM(payload)

    synth_alerts = generate_stage3_synthesis_issues(
        patient_context=ctx,
        deterministic_alerts=deterministic_alerts,
        evidence_map=evidence_map,
        llm_client=llm,  # type: ignore[arg-type]
    )

    assert synth_alerts, "Expected at least one synthesis Alert"
    a0 = synth_alerts[0]
    assert a0.evidence.get("type") == "synthesis_issue"
    issue = a0.evidence.get("issue")
    assert isinstance(issue, dict)
    for key in ("issue_type", "severity", "guideline_reference", "already_in_plan", "nudge_needed"):
        assert key in issue
