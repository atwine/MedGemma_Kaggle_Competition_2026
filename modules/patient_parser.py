"""Patient parsing utilities.

This module normalizes structured + unstructured patient data into a format
that deterministic alert rules can consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_iso_date(value: str) -> date:
    # Rationale: mock patient data uses ISO dates (YYYY-MM-DD). We normalize to `date`
    # for consistent comparisons.
    return datetime.strptime(value, "%Y-%m-%d").date()


def _days_between(d1: date, d2: date) -> int:
    return abs((d2 - d1).days)


@dataclass(frozen=True)
class LabResult:
    name: str
    date: date
    value: Any


@dataclass(frozen=True)
class PatientContext:
    patient_id: str
    name: str
    encounter_date: date
    art_regimen_current: List[str]
    notes_text: str
    latest_labs: Dict[str, LabResult]


def load_mock_patients(path: Path) -> List[Dict[str, Any]]:
    """Load demo patients from JSON."""

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_patient_context(patient: Dict[str, Any]) -> PatientContext:
    """Build a normalized patient context.

    The context includes:
    - encounter date (today)
    - current ART regimen
    - concatenated notes text (history + today)
    - latest lab values by lab name
    """

    encounter_date = _parse_iso_date(patient["today_encounter"]["date"])

    art_regimen_current = list(patient.get("art_regimen_current") or [])

    notes_parts: List[str] = []
    for v in patient.get("visits", []) or []:
        note = (v.get("clinician_note") or "").strip()
        if note:
            notes_parts.append(note)

    today_note = (patient.get("today_encounter", {}).get("note") or "").strip()
    if today_note:
        notes_parts.append(today_note)

    notes_text = "\n".join(notes_parts)

    latest_labs: Dict[str, LabResult] = {}
    labs = patient.get("labs") or {}

    # Rationale: keep lab naming stable across rules by using explicit keys.
    if "creatinine" in labs:
        latest = _latest_lab_entry(labs["creatinine"], encounter_date)
        if latest is not None:
            latest_labs["creatinine"] = latest

    if "viral_load" in labs:
        latest = _latest_lab_entry(labs["viral_load"], encounter_date)
        if latest is not None:
            latest_labs["viral_load"] = latest

    return PatientContext(
        patient_id=str(patient.get("patient_id")),
        name=str(patient.get("name")),
        encounter_date=encounter_date,
        art_regimen_current=art_regimen_current,
        notes_text=notes_text,
        latest_labs=latest_labs,
    )


def _latest_lab_entry(entries: List[Dict[str, Any]], encounter_date: date) -> Optional[LabResult]:
    # Rationale: choose the latest lab entry by date, ignoring any that are in the future
    # relative to the encounter date.
    parsed: List[LabResult] = []
    for e in entries or []:
        d = _parse_iso_date(e["date"])
        if d > encounter_date:
            continue

        # Use whichever value field exists; rules can interpret as needed.
        value = None
        for k, v in e.items():
            if k != "date":
                value = v
                break

        parsed.append(LabResult(name="", date=d, value=value))

    if not parsed:
        return None

    latest = max(parsed, key=lambda x: x.date)
    return latest


def days_since_lab(patient_context: PatientContext, lab_name: str) -> Optional[int]:
    """Return number of days since the latest lab, or None if missing."""

    lab = patient_context.latest_labs.get(lab_name)
    if lab is None:
        return None

    return _days_between(lab.date, patient_context.encounter_date)
