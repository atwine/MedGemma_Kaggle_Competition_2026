"""Deterministic alert rules.

This module is intentionally rule-based for demo reliability.
The LLM is used later for explanation/citation, not for deciding whether an alert triggers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from modules.patient_parser import PatientContext, days_since_lab


@dataclass(frozen=True)
class Alert:
    alert_id: str
    title: str
    message: str
    evidence: Dict[str, Any]
    query_hint: str


def run_alerts(context: PatientContext) -> List[Alert]:
    """Run all deterministic checks and return a list of alerts."""

    alerts: List[Alert] = []

    tdf_alert = _check_tdf_creatinine_overdue(context)
    if tdf_alert is not None:
        alerts.append(tdf_alert)

    efv_alert = _check_efv_sleep_disturbance(context)
    if efv_alert is not None:
        alerts.append(efv_alert)

    return alerts


def _check_tdf_creatinine_overdue(context: PatientContext) -> Optional[Alert]:
    """If on TDF, flag if creatinine is overdue.

    Threshold is set to 12 months for a conservative demo rule.
    """

    if "TDF" not in context.art_regimen_current:
        return None

    days = days_since_lab(context, "creatinine")
    if days is None:
        return Alert(
            alert_id="tdf_creatinine_missing",
            title="Creatinine missing while on TDF",
            message="Consider ordering creatinine/eGFR monitoring for a patient on TDF; no recent creatinine result found in history.",
            evidence={
                "regimen": context.art_regimen_current,
                "creatinine": None,
            },
            query_hint="TDF creatinine monitoring frequency",
        )

    if days <= 365:
        return None

    return Alert(
        alert_id="tdf_creatinine_overdue",
        title="Creatinine monitoring overdue while on TDF",
        message=f"Creatinine last recorded {days} days ago for a patient on TDF. Consider ordering repeat renal function monitoring.",
        evidence={
            "regimen": context.art_regimen_current,
            "days_since_creatinine": days,
        },
        query_hint="TDF renal function monitoring creatinine schedule",
    )


def _check_efv_sleep_disturbance(context: PatientContext) -> Optional[Alert]:
    """If on EFV, flag if notes mention sleep disturbance terms."""

    if "EFV" not in context.art_regimen_current:
        return None

    text = (context.notes_text or "").lower()
    keywords = [
        "sleep",
        "insomnia",
        "vivid dream",
        "vivid dreams",
        "nightmare",
    ]

    hits = [k for k in keywords if k in text]
    if not hits:
        return None

    return Alert(
        alert_id="efv_sleep_disturbance",
        title="Possible EFV neuropsychiatric side effects",
        message="Sleep disturbance mentioned in notes for a patient on EFV. Consider assessing EFV tolerance and reviewing guideline management options.",
        evidence={
            "regimen": context.art_regimen_current,
            "keyword_hits": hits,
        },
        query_hint="Efavirenz sleep disturbance vivid dreams management",
    )
