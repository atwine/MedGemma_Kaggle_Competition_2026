"""Patient parsing utilities.

This module normalizes structured + unstructured patient data into a format
that deterministic alert rules can consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    encounter_date: date
    art_regimen_current: List[str]
    notes_text: str
    latest_labs: Dict[str, LabResult]
    # NEW: Structured access to intake fields (optional, backward compatible)
    other_medications: List[str] = field(default_factory=list)
    complaints_symptoms: str = ""
    examination_findings: str = ""
    # NEW: Additional patient demographics and lab narrative
    age_years: Optional[int] = None
    sex: str = ""
    labs_narrative: str = ""


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
    - structured intake fields (optional, for new patient records)
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

    # NEW: Extract structured fields if available (backward compatible)
    other_meds = list(patient.get("other_medications") or [])
    intake = patient.get("intake") or {}
    complaints = intake.get("complaints_symptoms", "")
    exam_findings = intake.get("examination_findings", "")
    
    # Extract age and sex
    age_years = patient.get("age_years")
    sex = patient.get("sex", "")
    
    # Extract lab narrative text
    labs_narrative_parts: List[str] = []
    labs_raw = patient.get("labs") or {}
    if "labs_narrative" in labs_raw:
        for entry in labs_raw.get("labs_narrative", []):
            if isinstance(entry, dict):
                narrative = entry.get("narrative_text", "")
                if narrative:
                    labs_narrative_parts.append(narrative)
    labs_narrative = "\n\n".join(labs_narrative_parts)

    return PatientContext(
        patient_id=str(patient.get("patient_id")),
        encounter_date=encounter_date,
        art_regimen_current=art_regimen_current,
        notes_text=notes_text,
        latest_labs=latest_labs,
        other_medications=other_meds,
        complaints_symptoms=complaints,
        examination_findings=exam_findings,
        age_years=age_years,
        sex=sex,
        labs_narrative=labs_narrative,
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


# Lab trend computation -----------------------------------------------------


@dataclass(frozen=True)
class LabTrend:
    """Summary of a single lab's historical values and direction."""

    lab_name: str
    values: List[Dict[str, Any]]  # [{"date": "YYYY-MM-DD", "value": ...}, ...]
    direction: str  # "RISING", "FALLING", "STABLE", "SINGLE", "UNKNOWN"
    summary_text: str  # human-readable one-liner for LLM prompts


def _extract_numeric(entry: Dict[str, Any]) -> Optional[float]:
    """Pull the first numeric value from a lab entry, ignoring 'date'."""
    for k, v in entry.items():
        if k == "date":
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def compute_lab_trends(
    raw_labs: Dict[str, List[Dict[str, Any]]],
    encounter_date: Optional[date] = None,
) -> List[LabTrend]:
    """Compute trend direction and summary text for each lab in raw patient data.

    Rationale: doctors identify drug toxicity by spotting *trends* (e.g. rising
    creatinine over years) rather than a single value.  This helper converts
    the raw lab history into a structured summary the LLM can reason over.
    """

    trends: List[LabTrend] = []

    for lab_name, entries in (raw_labs or {}).items():
        if not entries:
            continue

        # Parse and sort chronologically, filtering out future dates if possible.
        parsed: List[Dict[str, Any]] = []
        for e in entries:
            try:
                d = _parse_iso_date(e["date"])
            except Exception:
                continue
            if encounter_date is not None and d > encounter_date:
                continue
            # Rationale: keep the original saved lab value text for display.
            raw_val = None
            for k, v in (e or {}).items():
                if k == "date":
                    continue
                raw_val = v
                break
            numeric = _extract_numeric(e)
            parsed.append({"date": e["date"], "value": numeric, "raw": raw_val, "date_obj": d})

        parsed.sort(key=lambda x: x["date_obj"])

        # Build the simplified values list (date + value only, no date_obj).
        values = [{"date": p["date"], "value": p["value"]} for p in parsed]

        # Determine direction from numeric values.
        numeric_vals = [p["value"] for p in parsed if p["value"] is not None]

        if len(numeric_vals) == 0:
            direction = "UNKNOWN"
        elif len(numeric_vals) == 1:
            direction = "SINGLE"
        else:
            # Compare each consecutive pair; count rises vs falls.
            rises = sum(1 for i in range(1, len(numeric_vals)) if numeric_vals[i] > numeric_vals[i - 1])
            falls = sum(1 for i in range(1, len(numeric_vals)) if numeric_vals[i] < numeric_vals[i - 1])
            pairs = len(numeric_vals) - 1
            if rises > pairs / 2:
                direction = "RISING"
            elif falls > pairs / 2:
                direction = "FALLING"
            else:
                direction = "STABLE"

        # Build a human-readable summary line.
        parts: List[str] = []
        for p in parsed:
            raw = p.get("raw")
            val = p.get("value")
            val_str = str(raw) if raw is not None else (str(val) if val is not None else "?")
            parts.append(f"{val_str} ({p['date']})")
        timeline = " -> ".join(parts) if parts else "no data"
        summary_text = f"{lab_name}: {timeline} — {direction}"

        trends.append(
            LabTrend(
                lab_name=lab_name,
                values=values,
                direction=direction,
                summary_text=summary_text,
            )
        )

    return trends
