from __future__ import annotations

from pathlib import Path

from modules.alert_rules import run_alerts
from modules.patient_parser import build_patient_context, load_mock_patients


def test_mock_patients_rule_expectations_smoke() -> None:
    data_path = Path(__file__).resolve().parents[1] / "Data" / "mock_patients.json"
    patients = load_mock_patients(data_path)

    # Rationale: this is a lightweight smoke test to ensure rules are wired correctly
    # to the demo data.
    outcomes = {}
    for p in patients:
        ctx = build_patient_context(p)
        alerts = run_alerts(ctx)
        outcomes[ctx.patient_id] = "YELLOW" if alerts else "GREEN"

    assert outcomes["P001"] == "YELLOW"
    assert outcomes["P002"] == "GREEN"
    assert outcomes["P003"] == "YELLOW"


def _base_patient(*, note: str, regimen: list[str] | None = None) -> dict:
    # Rationale: build a minimal synthetic patient record for focused rule tests.
    return {
        "patient_id": "TEST",
        "name": "Test Patient",
        "art_regimen_current": regimen or ["TDF", "3TC", "DTG"],
        "visits": [],
        "labs": {
            # Keep creatinine recent so the TDF creatinine overdue rule does not trigger.
            "creatinine": [{"date": "2026-01-01", "value_umol_per_l": 88}],
        },
        "today_encounter": {
            "date": "2026-01-22",
            "note": note,
            "orders": [],
            "med_changes": [],
        },
    }


def test_rule_tdf_nsaid_use_triggers_on_ibuprofen_mention() -> None:
    patient = _base_patient(note="Occasionally uses ibuprofen for back pain.")
    ctx = build_patient_context(patient)
    alerts = run_alerts(ctx)

    assert any(a.alert_id == "tdf_nsaid_use" for a in alerts)


def test_rule_possible_proteinuria_triggers_on_trace_protein_mention() -> None:
    patient = _base_patient(note="Trace protein noted on dipstick.")
    ctx = build_patient_context(patient)
    alerts = run_alerts(ctx)

    assert any(a.alert_id == "possible_proteinuria" for a in alerts)


def test_rule_suboptimal_bp_with_missed_amlodipine_triggers() -> None:
    patient = _base_patient(
        note=(
            "BP is sub-optimally controlled today; patient admits to occasionally missing Amlodipine."
        )
    )
    ctx = build_patient_context(patient)
    alerts = run_alerts(ctx)

    assert any(a.alert_id == "suboptimal_bp_missed_amlodipine" for a in alerts)
