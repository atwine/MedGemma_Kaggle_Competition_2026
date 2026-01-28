from __future__ import annotations

from modules.patient_parser import build_patient_context
from modules.stage1_summary import build_stage1_summary


def test_build_stage1_summary_structure_and_flags() -> None:
    patient = {
        "patient_id": "p123",
        "name": "Mr Test",
        "today_encounter": {
            "date": "2024-01-01",
            "note": "Plan: check creatinine and VL next visit.",
        },
        "visits": [
            {
                "date": "2023-12-01",
                "clinician_note": "Sleep disturbance lately.\nPlan: counsel sleep hygiene.",
            }
        ],
        "labs": {
            "creatinine": [{"date": "2023-01-01", "value_umol_per_l": 80}],
            "viral_load": [{"date": "2023-11-01", "value_copies_per_ml": 200}],
        },
    }

    ctx = build_patient_context(patient)
    summary = build_stage1_summary(patient=patient, context=ctx)

    # Top-level keys
    assert summary.get("patient_id") == "p123"
    assert "encounter_date" in summary
    assert isinstance(summary.get("current_regimen"), list)

    # Labs
    assert isinstance(summary.get("past_labs"), list)
    assert isinstance(summary.get("current_labs"), dict)
    # Today plan should capture lines starting with "Plan:"
    assert any(line.startswith("Plan:") for line in summary.get("today_plan", []))

    # Missing flags present
    mf = summary.get("missing_flags")
    assert isinstance(mf, dict)
    assert set(["regimen_history", "non_hiv_meds", "past_plans", "past_complaints", "today_plan"]).issubset(mf.keys())
