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
