"""Streamlit demo entrypoint for the HIV Clinical Nudge Engine.
 
 This app implements an initial local-first “Save vs Finalize” workflow:
 - Save runs deterministic alert rules + RAG retrieval (and optional Ollama explanation)
 - Finalize is blocked until each alert is acknowledged or overridden with a reason
 """
 
from __future__ import annotations

import copy
import importlib
import json
import os
import re
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import datetime

import streamlit as st

from modules.alert_rules import Alert, run_alerts
from modules.embedder import Embedder, EmbedderConfig
import modules.explanation_generator as explanation_generator

import html

# Rationale: Streamlit can keep stale module objects across reruns. If a newer
# export was added to `modules.explanation_generator`, force a reload only when
# the expected symbol is missing.
if not hasattr(explanation_generator, "generate_audit_checklist_alerts"):
    explanation_generator = importlib.reload(explanation_generator)

ExplanationResult = explanation_generator.ExplanationResult
generate_audit_checklist_alerts = explanation_generator.generate_audit_checklist_alerts
generate_explanation = explanation_generator.generate_explanation
generate_stage3_synthesis_issues = explanation_generator.generate_stage3_synthesis_issues
from modules.llm_client import OllamaClient, OllamaConfig
from modules.patient_parser import build_patient_context, load_mock_patients
from modules.rag_engine import RagEngine
from modules.vector_store import VectorSearchResult, create_vector_store
from modules.stage1_summary import build_stage1_summary
from modules.stage3_narrative import generate_stage3_narrative
from modules.llm_screening import generate_llm_screening_alerts
from modules.agentic_flow import QueryContext, run_agentic_flow


PROJECT_ROOT = Path(__file__).resolve().parent
CONSOLIDATED_GUIDELINE_PDF_PATH = PROJECT_ROOT / "Data" / "Consolidated-HIV-and-AIDS-Guidelines-20230516.pdf"
LEGACY_GUIDELINE_PDF_PATH = PROJECT_ROOT / "Data" / "Uganda Clinical Guidelines 2023.pdf"
NCD_GUIDELINE_PDF_PATH = (
    PROJECT_ROOT
    / "Data"
    / "Uganda+Integrated+Guidelines+for+the+management+of+NCDs-+Uganda+with+cover-FIN+14October2019.V4.pdf"
)

# Rationale: Markdown conversions of the guideline PDFs preserve table structures
# that pypdf extraction drops. We now embed from the Markdown files instead of PDFs.
GUIDELINE_MD_DIR = PROJECT_ROOT / "Data" / "MarkDown Files"
GUIDELINE_PATHS: list[Path] = sorted(
    p for p in GUIDELINE_MD_DIR.glob("*.md") if p.is_file()
) if GUIDELINE_MD_DIR.is_dir() else []

# Rationale: persistent Chroma collections keep old indexed chunks; bump this
# when the ingestion/cleaning/chunking logic changes to force a fresh collection.
GUIDELINE_INDEX_VERSION = 1
# Backward-compatible alias used by downstream code.
GUIDELINE_PDF_PATHS = GUIDELINE_PATHS
MOCK_PATIENTS_PATH = PROJECT_ROOT / "Data" / "mock_patients.json"
CUSTOM_PATIENTS_PATH = PROJECT_ROOT / "Data" / "custom_patients.json"
CANDIDATE_RULES_PATH = PROJECT_ROOT / "Data" / "candidate_rules.json"


@st.cache_data
def _load_patients(path: str, *, version: int = 1) -> List[Dict[str, Any]]:
    # Rationale: cache synthetic demo data across reruns for responsiveness.
    # The version parameter allows cache-busting when the JSON file changes.
    return load_mock_patients(Path(path))


def _resolve_ollama_host() -> str:
    """Resolve the Ollama host in the same way the Python client expects.

    Mirrors the logic in modules.llm_client.OllamaClient to avoid surprises when
    listing models vs running chat calls.
    """

    raw_host = (os.getenv("OLLAMA_HOST") or "").strip()
    if not raw_host:
        return "http://localhost:11434"
    host = raw_host
    if "://" not in host:
        host = f"http://{host}"
    host = host.replace("http://0.0.0.0", "http://localhost").replace(
        "https://0.0.0.0", "http://localhost"
    )
    return host


def _list_ollama_models() -> List[str]:
    """Return names of locally available Ollama models.

    Uses the official ollama-python Client.list() endpoint, which queries the
    /api/tags API under the hood. [src]
    - https://raw.githubusercontent.com/ollama/ollama/main/docs/api.md
    """

    try:
        from ollama import Client  # type: ignore[import]
    except Exception:
        return []

    try:
        client = Client(host=_resolve_ollama_host(), timeout=5.0)
        resp = client.list()
    except Exception:
        return []

    models: List[str] = []

    # Newer ollama-python may return an object with a `.models` attribute; older
    # versions or direct HTTP clients may return a dict or list. Handle all
    # three forms.
    items: Any
    if hasattr(resp, "models"):
        items = resp.models  # type: ignore[attr-defined]
    elif isinstance(resp, dict):
        items = resp.get("models") or resp.get("data") or []
    else:
        items = resp

    if isinstance(items, list):
        for item in items:
            # Object-style model (e.g., dataclass with `.name`).
            if hasattr(item, "name") and isinstance(getattr(item, "name"), str):
                name = getattr(item, "name")
                if name:
                    models.append(name)
                continue

            if isinstance(item, dict):
                name = item.get("name") or item.get("model")
                if isinstance(name, str) and name:
                    models.append(name)
            elif isinstance(item, str) and item:
                models.append(item)

    return models


def _load_custom_patients(path: Path) -> List[Dict[str, Any]]:
    try:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        out: List[Dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                out.append(item)
        return out
    except Exception:
        return []


def _save_custom_patients(path: Path, patients: List[Dict[str, Any]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(patients, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_candidate_rules(path: Path) -> List[Dict[str, Any]]:
    try:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        out: List[Dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                out.append(item)
        return out
    except Exception:
        return []


def _save_candidate_rules(path: Path, rules: List[Dict[str, Any]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
 
 
@st.cache_resource
def _get_embedder(model_name: str, *, version: int = 2) -> Embedder:
    # Rationale: embedding model load is expensive; cache the instance.
    return Embedder(EmbedderConfig(model_name=model_name))


@st.cache_resource
def _get_rag_engine(
    *,
    prefer_chroma: bool,
    embedding_model_name: str,
    engine_version: int = 3,
) -> RagEngine:
    # Rationale: keep a single vector store + embedder + engine per session.
    embedder = _get_embedder(embedding_model_name, version=2)
    # Rationale: keep a separate Chroma collection per embedding model to avoid
    # mixing incompatible vector spaces across runs.
    model_key = (
        (embedding_model_name or "")
        .strip()
        .lower()
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )

    # Rationale: persistent Chroma collections keep old indexed chunks. Include the
    # active guideline sources in the collection name to prevent mixing chunks from
    # previous documents and to force a clean re-index when the source files change.
    guideline_key = "no_guideline"
    if GUIDELINE_PATHS:
        # Rationale: hash all file stems so the collection auto-resets when any
        # source file is added, removed or renamed.
        guideline_key = "_".join(
            p.stem.strip().lower().replace(" ", "_").replace("+", "_")[:20]
            for p in GUIDELINE_PATHS
        )[:80]  # keep the collection name within Chroma's limits
    vector_store = create_vector_store(
        project_root=PROJECT_ROOT,
        prefer_chroma=prefer_chroma,
        collection_name=f"uganda_hiv_guidelines__{model_key}__{guideline_key}__v{GUIDELINE_INDEX_VERSION}",
    )
    return RagEngine(
        project_root=PROJECT_ROOT,
        guideline_paths=GUIDELINE_PATHS,
        embedder=embedder,
        vector_store=vector_store,
    )


def _init_session_state() -> None:
    # Rationale: Streamlit reruns on every interaction; initialize defaults once.
    st.session_state.setdefault("analysis_ran", False)
    st.session_state.setdefault("finalized", False)
    st.session_state.setdefault("analysis_status", None)
    st.session_state.setdefault("analysis_alerts", [])
    st.session_state.setdefault("analysis_results", [])
    st.session_state.setdefault("analysis_use_ollama", False)
    st.session_state.setdefault("analysis_ollama_model", None)
    st.session_state.setdefault("analysis_ollama_error", None)
    # Rationale: allow audit/debug display for LLM screening when no alerts are returned.
    st.session_state.setdefault("analysis_screening_debug", None)
    st.session_state.setdefault("analysis_screening_patient_facts", None)
    # Rationale: allow adding ad-hoc test cases without editing JSON files.
    st.session_state.setdefault("custom_patients", [])


def _extract_patient_facts_from_history(notes_text: str) -> Dict[str, Any]:
    # Rationale: deterministic, lightweight extraction so clinicians can see what the system
    # believes is present in the free-text history even when no alerts are generated.
    text = (notes_text or "").strip()
    low = text.lower()

    # Rationale: match common ART abbreviations and full names to reduce false "empty" fact extraction.
    # This does not change alert logic; it only improves audit transparency.
    arv_terms = {
        "tdf": "TDF",
        "tenofovir": "TDF",
        "3tc": "3TC",
        "lamivudine": "3TC",
        "dtg": "DTG",
        "dolutegravir": "DTG",
        "azt": "AZT",
        "zidovudine": "AZT",
        "ftc": "FTC",
        "emtricitabine": "FTC",
        "efv": "EFV",
        "efavirenz": "EFV",
        "nvp": "NVP",
        "nevirapine": "NVP",
        "d4t": "D4T",
        "stavudine": "D4T",
        "abc": "ABC",
        "abacavir": "ABC",
        "taf": "TAF",
    }
    meds_norm = set()
    for term, canon in arv_terms.items():
        if re.search(rf"\b{re.escape(term)}\b", low):
            meds_norm.add(canon)
    meds_found = sorted(meds_norm)

    symptom_terms = [
        "bone pain",
        "fracture",
        "difficulty walking",
        "proximal muscle weakness",
        "weakness",
        "fatigue",
        "dizziness",
    ]
    symptoms_found = [t for t in symptom_terms if t in low]

    key_line_terms = [
        "current art",
        "viral load",
        "cd4",
        "serum creatinine",
        "creatinine",
        "egfr",
        "urinalysis",
        "phosphate",
        "alp",
        "vitamin d",
        "dexa",
        "t-score",
        "looser",
    ]
    key_lines: List[str] = []
    for ln in (text.splitlines() or []):
        s = ln.strip()
        if not s:
            continue
        sl = s.lower()
        if any(term in sl for term in key_line_terms):
            key_lines.append(s)

    # Bound the output so it remains readable in the UI.
    return {
        # Rationale: include minimal diagnostics so a clinician can confirm what text was actually checked.
        "notes_text_char_count": len(text),
        "notes_text_preview": (text[:300] + ("…" if len(text) > 300 else "")),
        "medications_detected": meds_found,
        "symptoms_detected": symptoms_found,
        "key_history_lines": key_lines[:25],
    }


def _compute_overall_status(alerts: List[Alert]) -> str:
    if not alerts:
        return "GREEN"

    # Rationale: if severity is available (e.g., from LLM screening issues), use the
    # worst severity to drive the overall status colour. Deterministic alerts that
    # do not specify a severity are treated as at least moderate.
    severity_rank = {"low": 0, "moderate": 1, "high": 2, "critical": 3}
    max_rank = 1  # default to "moderate" when any alert exists but no severity set
    for alert in alerts:
        evidence = alert.evidence or {}
        sev = str(evidence.get("severity") or "").lower()
        rank = severity_rank.get(sev, max_rank)
        if rank > max_rank:
            max_rank = rank

    if max_rank >= 2:
        return "RED"
    return "YELLOW"


def _alert_resolution_ok(alert_id: str) -> bool:
    action = st.session_state.get(f"alert_action_{alert_id}")
    if action == "Acknowledge":
        return True

    if action == "Override":
        reason = (st.session_state.get(f"alert_override_reason_{alert_id}") or "").strip()
        return bool(reason)

    return False


def _can_finalize(alerts: List[Alert]) -> bool:
    if not st.session_state.get("analysis_ran", False):
        return False

    if not alerts:
        return True

    return all(_alert_resolution_ok(a.alert_id) for a in alerts)


def _run_analysis(
    *,
    patient: Dict[str, Any],
    today_note: str,
    prefer_chroma: bool,
    embedding_model_name: str,
    top_k: int,
    index_max_pages: Optional[int],
    retrieval_page_range: Optional[tuple[int, int]],
    use_ollama: bool,
    llm_mode: str,
    ollama_model: Optional[str],
    ollama_num_ctx: Optional[int],
    show_llm_narrative: bool,
) -> None:
    # Rationale: keep analysis deterministic for alert triggering, and only use the
    # LLM for explanation/citation if available.

    patient_copy = copy.deepcopy(patient)
    patient_copy.setdefault("today_encounter", {})
    patient_copy["today_encounter"]["note"] = today_note

    context = build_patient_context(patient_copy)
    alerts = run_alerts(context)
    # Preserve Stage 1 deterministic alerts for Stage 3 synthesis input.
    deterministic_alerts = list(alerts)
    # Build Stage 1 summary once so it can be reused by LLM screening and narrative.
    stage1_summary = build_stage1_summary(patient=patient_copy, context=context)

    rag = _get_rag_engine(
        prefer_chroma=prefer_chroma,
        embedding_model_name=embedding_model_name,
        # Rationale: cache-bust to refresh vector_store Chroma filter fix and related changes.
        engine_version=5,
    )

    # Rationale: indexing can be expensive; we allow limiting pages for a faster demo.
    if GUIDELINE_PDF_PATHS:
        rag.ensure_indexed(max_pages=index_max_pages)

    llm_client = None
    if use_ollama:
        cfg = OllamaConfig(model=(ollama_model or "aadide/medgemma-1.5-4b-it-Q4_K_S"), num_ctx=ollama_num_ctx)
        llm_client = OllamaClient(cfg)

    # Optional: agentic planner + per-subtask retrieval (debug-only, no UI change).
    env_flag = (os.getenv("AGENTIC_RAG_DEBUG") or "").strip() == "1"
    ui_flag = bool(st.session_state.get("agentic_ui_debug_enabled"))
    agentic_debug_enabled = bool(env_flag or ui_flag)
    agentic_debug_result = None

    if agentic_debug_enabled:
        try:
            agentic_context = QueryContext(
                question_text=(
                    "Review this patient's symptoms, examination findings, "
                    "and lab results (including trends over time). Could any "
                    "of them be signs of drug toxicity from their current or "
                    "past ART medications or co-medications? If so, what "
                    "tests confirm it, and does the doctor's current plan "
                    "adequately address the issue per Uganda HIV guidelines?"
                ),
                patient_raw=patient_copy,
                patient_summary=stage1_summary,
                metadata={
                    "llm_mode": llm_mode,
                    "top_k": top_k,
                    "index_max_pages": index_max_pages,
                    "retrieval_page_range": retrieval_page_range,
                },
            )

            agentic_debug_result = run_agentic_flow(
                agentic_context,
                embedder=rag._embedder,
                vector_store=rag._vector_store,
                top_k=top_k,
                page_range=retrieval_page_range,
                llm_client=llm_client,
            )

        except Exception as exc:
            agentic_debug_result = {
                "error": str(exc),
                "type": exc.__class__.__name__,
            }

    st.session_state["agentic_debug_enabled"] = agentic_debug_enabled
    st.session_state["agentic_debug_result"] = agentic_debug_result

    # Status box for any LLM activity in this run.
    ollama_status = st.status("Ollama: idle", expanded=False) if use_ollama else None

    ollama_error: Optional[str] = None

    # Rationale: LLM-first audit checklist layer (Option B) - checklist items are
    # represented as Alert objects and must be acknowledged/overridden before Finalize.
    checklist_seed = Alert(
        alert_id="llm_checklist_seed",
        title="Guideline audit checklist",
        message="LLM audit checklist seed",
        evidence={"type": "llm_audit_checklist_seed"},
        query_hint="visit audit checklist labs monitoring regimen",
    )
    # Rationale: compatibility — during hot-reload, a cached RagEngine may not yet
    # accept the new page_range kw. Try new signature first, then fall back.
    try:
        checklist_retrieved = rag.retrieve_for_alert(
            patient_context=context,
            alert=checklist_seed,
            top_k=top_k,
            page_range=retrieval_page_range,
        )
    except (TypeError, ValueError):
        checklist_retrieved = rag.retrieve_for_alert(
            patient_context=context,
            alert=checklist_seed,
            top_k=top_k,
        )
    # Client-side guard: enforce page-range bounds post-retrieval to avoid any caching/store drift.
    if retrieval_page_range is not None:
        lo, hi = retrieval_page_range
        checklist_retrieved = [
            r for r in checklist_retrieved
            if (r.metadata.get("page_number") is not None and int(r.metadata.get("page_number")) >= int(lo) and int(r.metadata.get("page_number")) <= int(hi))
        ]

    if ollama_status is not None:
        # Rationale: the first LLM call is a good proxy for "Ollama connected".
        ollama_status.update(label="Ollama: generating audit checklist…", state="running")
    # Manage LLM calls by mode: checklist vs per‑alert explanations vs synthesis vs screening.
    # In Synthesis mode, skip the checklist LLM to keep a single LLM call later.
    checklist_llm = llm_client if (use_ollama and llm_mode == "Checklist only") else None
    explanation_llm = llm_client if (use_ollama and llm_mode == "Per-alert explanations") else None

    checklist_alerts: List[Alert] = []
    if llm_mode == "Checklist only":
        if ollama_status is not None and checklist_llm is not None:
            ollama_status.update(label="Ollama: generating audit checklist…", state="running")
        checklist_alerts = generate_audit_checklist_alerts(
            patient_context=context,
            retrieved_chunks=checklist_retrieved,
            llm_client=checklist_llm,
        )

        if ollama_status is not None and checklist_llm is not None:
            last_err = checklist_llm.last_error if checklist_llm is not None else None
            if last_err:
                ollama_status.update(label=f"Ollama error: {last_err}", state="error")
            else:
                ollama_status.update(label="Ollama connected (response received)", state="complete")
        if checklist_alerts:
            alerts = list(alerts) + checklist_alerts

    # Optional: LLM-first screening mode — generate alerts directly from Stage 1 summary
    # and checklist evidence. Deterministic alerts remain the fallback when
    # screening fails (error or unparseable output).
    if (
        use_ollama
        and llm_mode == "LLM screening (generate alerts)"
        and llm_client is not None
    ):
        if ollama_status is not None:
            ollama_status.update(label="Ollama: screening for safety issues…", state="running")
        screening_debug: Dict[str, Any] = {}
        screening_alerts = generate_llm_screening_alerts(
            patient_context=context,
            stage1_summary=stage1_summary,
            screening_chunks=checklist_retrieved,
            llm_client=llm_client,
            debug=screening_debug,
        )
        # Rationale: persist screening audit inputs/parse status for GREEN "no alerts" explanation.
        st.session_state["analysis_screening_debug"] = screening_debug
        st.session_state["analysis_screening_patient_facts"] = _extract_patient_facts_from_history(
            context.notes_text
        )

        parse_status = (screening_debug.get("parse_status") or "").strip().lower()
        last_err = llm_client.last_error

        screening_succeeded = (
            last_err is None
            and parse_status in {"parsed_empty", "parsed_nonempty"}
        )

        if screening_succeeded:
            alerts = list(screening_alerts)
            deterministic_alerts = []

        if ollama_status is not None:
            if last_err:
                ollama_status.update(label=f"Ollama error: {last_err}", state="error")
            else:
                ollama_status.update(label="Ollama connected (response received)", state="complete")
        if ollama_error is None:
            ollama_error = last_err

    # Rationale: reset per-alert review state for each analysis run. Without this,
    # a previous acknowledgment/override can incorrectly allow finalize on a new
    # Save without review.
    for alert in alerts:
        st.session_state[f"alert_action_{alert.alert_id}"] = "Unreviewed"
        st.session_state[f"alert_override_reason_{alert.alert_id}"] = ""
        st.session_state[f"alert_override_comment_{alert.alert_id}"] = ""

    results: List[Dict[str, Any]] = []
    # Evidence map for Stage 3 synthesis (per deterministic alert).
    evidence_map: Dict[str, List[VectorSearchResult]] = {}

    for alert in alerts:
        if (alert.evidence or {}).get("type") == "llm_audit_checklist":
            # Rationale: checklist alerts already contain LLM-produced guidance and
            # citations; avoid additional LLM calls per checklist item.
            retrieved: List[VectorSearchResult] = []
            explanation = ExplanationResult(
                text=alert.message,
                used_chunks=[],
                used_llm=False,
            )
        else:
            try:
                retrieved = rag.retrieve_for_alert(
                    patient_context=context,
                    alert=alert,
                    top_k=top_k,
                    # Rationale: Stage 2 — propagate page range filter for per-alert retrieval.
                    page_range=retrieval_page_range,
                )
            except (TypeError, ValueError):
                retrieved = rag.retrieve_for_alert(
                    patient_context=context,
                    alert=alert,
                    top_k=top_k,
                )
            # Client-side guard: enforce page-range bounds post-retrieval to avoid any caching/store drift.
            if retrieval_page_range is not None:
                lo, hi = retrieval_page_range
                retrieved = [
                    r for r in retrieved
                    if (r.metadata.get("page_number") is not None and int(r.metadata.get("page_number")) >= int(lo) and int(r.metadata.get("page_number")) <= int(hi))
                ]

            # Capture evidence for Stage 3 only for deterministic (non-LLM) alerts.
            evidence_map[alert.alert_id] = list(retrieved)

            if ollama_status is not None and explanation_llm is not None:
                # Rationale: per-alert LLM calls may each take time; show which step is active.
                ollama_status.update(
                    label=f"Ollama: generating explanation for '{alert.title}'…",
                    state="running",
                )
            # Rationale: measure per‑alert LLM explanation time to track latency and surface it in the UI.
            _t0 = time.perf_counter()
            explanation = generate_explanation(
                patient_context=context,
                alert=alert,
                retrieved_chunks=retrieved,
                llm_client=explanation_llm,
            )
            _t1 = time.perf_counter()
            st.session_state[f"explain_time_{alert.alert_id}"] = max(0.0, float(_t1 - _t0))

            if ollama_status is not None and explanation_llm is not None:
                last_err = explanation_llm.last_error if explanation_llm is not None else None
                if last_err:
                    ollama_status.update(label=f"Ollama error: {last_err}", state="error")
                else:
                    ollama_status.update(label="Ollama connected (response received)", state="complete")

        # Rationale: capture the first LLM failure reason so the UI can show why
        # it fell back to deterministic mode.
        if explanation_llm is not None and not explanation.used_llm:
            last_err = explanation_llm.last_error
            if last_err and ollama_error is None:
                ollama_error = last_err

        results.append(
            {
                "alert": alert,
                "retrieved": retrieved,
                "explanation": explanation,
            }
        )

    # Clinical summary (LLM) feature disabled: always clear any prior narrative.
    st.session_state["analysis_narrative"] = None

    # Stage 3 synthesis: single LLM call producing structured issues as new Alerts.
    if use_ollama and llm_mode == "Synthesis" and llm_client is not None:
        if ollama_status is not None:
            ollama_status.update(label="Ollama: generating synthesis issues…", state="running")
        try:
            synth_alerts = generate_stage3_synthesis_issues(
                patient_context=context,
                deterministic_alerts=deterministic_alerts,
                evidence_map=evidence_map,
                llm_client=llm_client,
            )
        except Exception:
            synth_alerts = []
        # Initialize gating state for new synthesis alerts and append to results.
        if synth_alerts:
            for a in synth_alerts:
                st.session_state[f"alert_action_{a.alert_id}"] = "Unreviewed"
                st.session_state[f"alert_override_reason_{a.alert_id}"] = ""
                st.session_state[f"alert_override_comment_{a.alert_id}"] = ""
                results.append(
                    {
                        "alert": a,
                        "retrieved": [],
                        "explanation": ExplanationResult(text=a.message, used_chunks=[], used_llm=False),
                    }
                )
            alerts = list(alerts) + synth_alerts

        if ollama_status is not None:
            last_err = llm_client.last_error
            if last_err:
                ollama_status.update(label=f"Ollama error: {last_err}", state="error")
            else:
                ollama_status.update(label="Ollama connected (response received)", state="complete")
        # Capture first error if no LLM text returned.
        if ollama_error is None:
            ollama_error = llm_client.last_error

    st.session_state["analysis_ran"] = True
    st.session_state["finalized"] = False
    st.session_state["analysis_alerts"] = alerts
    st.session_state["analysis_results"] = results
    st.session_state["analysis_status"] = _compute_overall_status(alerts)
    st.session_state["analysis_use_ollama"] = bool(use_ollama)
    st.session_state["analysis_llm_mode"] = llm_mode
    # UX: surface active retrieval page range in Results view (caption) for reviewers.
    st.session_state["analysis_retrieval_page_range"] = retrieval_page_range
    _client_used = (checklist_llm or explanation_llm)
    st.session_state["analysis_ollama_model"] = _client_used.model if _client_used is not None else None
    st.session_state["analysis_ollama_error"] = ollama_error

    agentic_reasoner_debug = None
    if st.session_state.get("agentic_debug_enabled"):
        try:
            base_result = st.session_state.get("agentic_debug_result")
            agentic_reasoner_debug = {
                "final_status": st.session_state.get("analysis_status"),
                "final_alert_count": len(alerts),
                "deterministic_alert_ids": [a.alert_id for a in deterministic_alerts],
                "final_alert_ids": [a.alert_id for a in alerts],
                "evidence_map_keys": list(evidence_map.keys()),
                "agentic_result_present": base_result is not None,
            }
        except Exception as exc:
            agentic_reasoner_debug = {
                "error": str(exc),
                "type": exc.__class__.__name__,
            }
    st.session_state["agentic_debug_reasoner"] = agentic_reasoner_debug


def main() -> None:
    st.set_page_config(page_title="HIV Clinical Nudge Engine", layout="wide")
    _init_session_state()

    # Rationale: avoid modifying widget-backed session_state keys after widgets are
    # instantiated. We clear the Add patient case form on the next rerun.
    if st.session_state.pop("reset_add_case_form", False):
        for _k in [
            "new_case_patient_id",
            "new_case_patient_name",
            "new_case_age",
            "new_case_sex",
            "new_case_art_regimen",
            "new_case_other_meds",
            "new_case_complaints",
            "new_case_exam_findings",
            "new_case_prior_history",
            "new_case_lab_results",
        ]:
            st.session_state.pop(_k, None)

    _flash = st.session_state.pop("add_case_flash_success", None)
    if isinstance(_flash, str) and _flash.strip():
        st.success(_flash)

    st.title("HIV Clinical Nudge Engine")
    st.caption(
        "Decision support only. Clinician retains final authority. Synthetic data only."
    )

    if not MOCK_PATIENTS_PATH.exists():
        st.error("Mock patients file not found: `Data/mock_patients.json`")
        return

    # Cache-bust: ensure edits to Data/mock_patients.json appear without manual cache clear.
    patients = _load_patients(str(MOCK_PATIENTS_PATH), version=2)
    if not patients:
        st.error("No mock patients found in `Data/mock_patients.json`")
        return

    custom_session = list(st.session_state.get("custom_patients") or [])
    custom_disk = _load_custom_patients(CUSTOM_PATIENTS_PATH)
    merged_custom: List[Dict[str, Any]] = []
    seen_ids = set()
    for p in custom_disk + custom_session:
        pid = p.get("patient_id")
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            merged_custom.append(p)
    custom_patients = merged_custom
    st.session_state["custom_patients"] = custom_patients

    with st.expander("Add patient case", expanded=False):
        st.markdown("""
            <div style='background-color: #FFF4E6; padding: 12px; border-radius: 6px; border-left: 4px solid #F4A261; margin-bottom: 15px;'>
                <p style='color: #E76F51; margin: 0; font-weight: 500;'>⚠️ For testing: avoid entering real names/identifiers. Use coded IDs or de-identified data.</p>
            </div>
        """, unsafe_allow_html=True)

        new_patient_id = st.text_input("Patient Identifier", key="new_case_patient_id")

        col1, col2 = st.columns(2)
        with col1:
            new_age = st.number_input(
                "Age (years)",
                min_value=0,
                max_value=120,
                value=None,
                key="new_case_age",
                help="Patient age in years"
            )
        with col2:
            new_sex = st.selectbox(
                "Sex",
                options=["", "M", "F"],
                key="new_case_sex",
                help="Biological sex"
            )

        new_art_regimen = st.text_input(
            "Current ART Regimen (comma-separated, e.g., TDF, 3TC, DTG)",
            key="new_case_art_regimen",
            help="Enter current antiretroviral medications separated by commas"
        )

        new_other_meds = st.text_area(
            "Other Medications",
            key="new_case_other_meds",
            height=60,
            help="Non-ARV medications (e.g., ibuprofen, metformin, carbamazepine). One per line or comma-separated."
        )

        new_complaints = st.text_area(
            "Complaints & Symptoms",
            key="new_case_complaints",
            height=80,
            help="Current symptoms and patient complaints"
        )

        new_exam_findings = st.text_area(
            "Examination Findings",
            key="new_case_exam_findings",
            height=80,
            help="Physical examination findings and observations"
        )

        st.caption("Optional: enter prior labs, events, and historical notes.")
        new_prior_history = st.text_area(
            "Significant history prior to today's visit",
            key="new_case_prior_history",
            height=80
        )

        st.caption("Optional: enter laboratory results (one result per line).")
        new_todays_lab_results = st.text_area(
            "Today's laboratory results",
            key="new_case_todays_lab_results",
            height=120,
            help=(
                "Enter today's lab results. Supported examples:\n"
                "viral_load, 2025-10-01, 40\n"
                "creatinine, 2024-10-01, 88"
            ),
        )
        
        new_previous_lab_results = st.text_area(
            "Previous Laboratory results",
            key="new_case_previous_lab_results",
            height=120,
            help=(
                "Enter previous lab results. Supported examples:\n"
                "viral_load, 2025-10-01, 40\n"
                "creatinine, 2024-10-01, 88"
            ),
        )

        if st.button("Add case to patient list"):
            pid = (new_patient_id or "").strip()
            if not pid:
                st.error("Patient ID is required.")
            elif any((p.get("patient_id") == pid) for p in (list(patients) + custom_patients)):
                st.error("That Patient ID already exists in the current patient list.")
            else:
                encounter_iso = datetime.date.today().isoformat()

                def _parse_bulk_labs(raw_text: str) -> Dict[str, List[Dict[str, Any]]]:
                    # Rationale: accept copy/paste and persist the same schema as Data/mock_patients.json.
                    # Supported examples:
                    # viral_load, 2025-10-01, 40
                    # creatinine, 2024-10-01, 88
                    # Narrative paste is also supported (e.g., "Creatinine: 1.4 mg/dL ... Viral load: <50").
                    out: Dict[str, List[Dict[str, Any]]] = {}
                    errors: List[str] = []
                    narrative_lines: List[str] = []

                    iso_date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")

                    for idx, ln in enumerate((raw_text or "").splitlines(), start=1):
                        s = (ln or "")
                        if not s.strip():
                            continue

                        # Only treat as structured CSV if the 2nd token is a valid ISO date.
                        # Rationale: preserve the lab value exactly as typed (no numeric conversion,
                        # no removal of symbols like '<' / '>', no unit conversion).
                        parts = s.split(",", 2)
                        if len(parts) == 3 and iso_date_re.match(parts[1].strip() or ""):
                            lab_raw, date_str, value_str = parts[0], parts[1].strip(), parts[2]
                            try:
                                datetime.date.fromisoformat(date_str)
                            except Exception:
                                errors.append(f"Labs: line {idx} has invalid date '{date_str}'")
                                continue

                            low = (lab_raw or "").strip().lower().replace(" ", "_")
                            if low in {"viral_load", "viral", "vl"}:
                                lab_name = "viral_load"
                                value_key = "value_copies_per_ml"
                            elif low in {"creatinine", "creat", "cr"}:
                                lab_name = "creatinine"
                                value_key = "value_umol_per_l"
                            else:
                                lab_name = (lab_raw or "") or low
                                value_key = "value"

                            out.setdefault(lab_name, []).append({"date": date_str, value_key: value_str})
                            continue

                        # Non-structured line: treat as narrative text; we'll extract key labs via regex below.
                        narrative_lines.append(s)

                    # If the user pasted narrative labs (single paragraph), save entire text as-is.
                    # Rationale: preserve ALL lab data exactly as entered, including historical values,
                    # additional tests, and context that would be lost by regex extraction.
                    if narrative_lines:
                        narrative_text = " ".join(narrative_lines)
                        default_date = encounter_iso
                        
                        # Save the complete narrative text under a generic "labs_narrative" key
                        # so nothing is lost and the doctor can see exactly what was entered.
                        out.setdefault("labs_narrative", []).append(
                            {"date": default_date, "narrative_text": narrative_text}
                        )

                    if errors:
                        raise ValueError("; ".join(errors))
                    return out

                # Parse both today's and previous lab results
                labs_parsed = {}
                try:
                    if new_todays_lab_results:
                        todays_labs = _parse_bulk_labs(new_todays_lab_results)
                        # Merge today's labs into labs_parsed with source tag
                        for lab_name, entries in todays_labs.items():
                            if lab_name == "labs_narrative":
                                # Tag narrative entries as "today"
                                for entry in entries:
                                    entry["source"] = "today"
                            labs_parsed.setdefault(lab_name, []).extend(entries)
                except ValueError as exc:
                    st.error(f"Error in today's lab results: {str(exc)}")
                    st.stop()
                
                try:
                    if new_previous_lab_results:
                        previous_labs = _parse_bulk_labs(new_previous_lab_results)
                        # Merge previous labs into labs_parsed with source tag
                        for lab_name, entries in previous_labs.items():
                            if lab_name == "labs_narrative":
                                # Tag narrative entries as "previous"
                                for entry in entries:
                                    entry["source"] = "previous"
                            labs_parsed.setdefault(lab_name, []).extend(entries)
                except ValueError as exc:
                    st.error(f"Error in previous lab results: {str(exc)}")
                    st.stop()

                # Parse ART regimen from comma-separated input
                # Rationale: preserve exact text as entered (no .strip())
                art_regimen = []
                if new_art_regimen:
                    art_regimen = [
                        drug
                        for drug in new_art_regimen.split(",")
                        if drug
                    ]

                # Parse other medications (support both newline and comma-separated)
                # Rationale: preserve exact text as entered (no .strip())
                other_meds = []
                if new_other_meds:
                    other_meds_text = (new_other_meds or "")
                    # Try newline-separated first, then comma-separated
                    if "\n" in other_meds_text:
                        other_meds = [
                            med
                            for med in other_meds_text.split("\n")
                            if med
                        ]
                    else:
                        other_meds = [
                            med
                            for med in other_meds_text.split(",")
                            if med
                        ]

                patient_record: Dict[str, Any] = {
                    "patient_id": pid,
                    "age_years": new_age if new_age is not None else None,
                    "sex": new_sex if new_sex else None,
                    "art_regimen_current": art_regimen,
                    "other_medications": other_meds,
                    "visits": [],
                    "labs": labs_parsed,
                    "today_encounter": {
                        "date": encounter_iso,
                        "note": "",
                        "orders": [],
                        "med_changes": [],
                    },
                    "intake": {
                        "complaints_symptoms": (new_complaints or ""),
                        "examination_findings": (new_exam_findings or ""),
                        "prior_history": (new_prior_history or ""),
                    },
                }

                # BACKWARD COMPATIBILITY: Concatenate structured fields into legacy visits[] format
                # so existing code (build_patient_context, RAG, LLM prompts) continues to work
                legacy_note_parts = []
                if patient_record["intake"]["complaints_symptoms"]:
                    legacy_note_parts.append(
                        f"Complaints & Symptoms:\n{patient_record['intake']['complaints_symptoms']}"
                    )
                if patient_record["intake"]["examination_findings"]:
                    legacy_note_parts.append(
                        f"Examination Findings:\n{patient_record['intake']['examination_findings']}"
                    )
                if patient_record["intake"]["prior_history"]:
                    legacy_note_parts.append(
                        f"Significant history prior to today's visit:\n{patient_record['intake']['prior_history']}"
                    )

                if legacy_note_parts:
                    patient_record["visits"].append(
                        {
                            "date": encounter_iso,
                            "type": "intake",
                            "clinician_note": "\n\n".join(legacy_note_parts),
                        }
                    )

                # Rationale: keep the local list in sync so the newly added patient
                # appears in the list immediately on this rerun.
                custom_patients = custom_patients + [patient_record]
                st.session_state["custom_patients"] = custom_patients
                _save_custom_patients(CUSTOM_PATIENTS_PATH, custom_patients)
                # Rationale: auto-select the newly added patient to make it clear
                # the add succeeded.
                st.session_state["selected_patient_label"] = pid
                # Rationale: clear the Add patient case form after success to allow rapid entry.
                st.session_state["add_case_flash_success"] = "Case added."
                st.session_state["reset_add_case_form"] = True
                st.rerun()

    if custom_patients:
        # Rationale: keep demo patients first, then any manually-added cases.
        patients = list(patients) + custom_patients

    st.subheader("Encounter")

    patient_labels = [f"{p.get('patient_id')} - {p.get('name')}" for p in patients]
    # Rationale: persist patient selection across reruns, and safely reset if the
    # selected value no longer exists (e.g., list changed).
    if (
        st.session_state.get("selected_patient_label") is None
        or st.session_state.get("selected_patient_label") not in patient_labels
    ):
        st.session_state["selected_patient_label"] = patient_labels[0]

    selected_label = st.selectbox(
        "Select patient",
        patient_labels,
        key="selected_patient_label",
    )
    selected_index = patient_labels.index(selected_label)
    patient = patients[selected_index]

    selected_patient_id = str(patient.get("patient_id") or "")
    custom_patient_ids = {str(p.get("patient_id") or "") for p in (custom_patients or [])}
    can_delete_selected = bool(selected_patient_id) and selected_patient_id in custom_patient_ids

    # Display Patient Details/History in expandable section
    with st.expander("Patient Details/History", expanded=False):
        # Helper function to calculate regimen duration
        def _calculate_regimen_duration(regimen_history):
            if not regimen_history or len(regimen_history) == 0:
                return None
            current_regimen = regimen_history[-1]
            start_date_str = current_regimen.get("start_date")
            if not start_date_str:
                return None
            try:
                from datetime import datetime
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
                today = datetime.now()
                years = (today - start_date).days / 365.25
                return years
            except Exception:
                return None

        # Create 2-column layout for compact display
        col1, col2 = st.columns(2)
        
        with col1:
            # Section 1: Patient Identifier & Visit Context
            st.markdown("""
                <div style='background-color: #E8F4F8; padding: 15px; border-radius: 8px; border-left: 4px solid #2E86AB; margin-bottom: 15px;'>
                    <h4 style='color: #2E86AB; margin-top: 0;'>1. Patient Identifier & Visit Context</h4>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"**ID Number:** {patient.get('patient_id', 'Not recorded')}")
            age = patient.get('age_years')
            st.markdown(f"**Age:** {age if age is not None else 'Not recorded'}")
            sex = patient.get('sex')
            st.markdown(f"**Sex:** {sex if sex else 'Not recorded'}")
            visit_date = patient.get('today_encounter', {}).get('date', 'Not recorded')
            st.markdown(f"**Visit Date:** {visit_date}")
            st.markdown(f"**Visit Type:** Routine")
            st.markdown("")

            # Section 2: Current ART Regimen
            st.markdown("""
                <div style='background-color: #E8F4F8; padding: 15px; border-radius: 8px; border-left: 4px solid #2E86AB; margin-bottom: 15px;'>
                    <h4 style='color: #2E86AB; margin-top: 0;'>2. Current ART Regimen</h4>
                </div>
            """, unsafe_allow_html=True)
            art_regimen = patient.get('art_regimen_current', [])
            if art_regimen:
                regimen_str = ", ".join(art_regimen)
                st.markdown(f"**Regimen:** <span style='color: #2E86AB; font-weight: 600;'>{regimen_str}</span>", unsafe_allow_html=True)
            else:
                st.markdown("**Regimen:** None recorded")
            st.markdown("")

            # Section 3: Other Medications
            st.markdown("""
                <div style='background-color: #F0F8F0; padding: 15px; border-radius: 8px; border-left: 4px solid #52B788; margin-bottom: 15px;'>
                    <h4 style='color: #52B788; margin-top: 0;'>3. Other Medications</h4>
                </div>
            """, unsafe_allow_html=True)
            other_meds = patient.get('other_medications', [])
            if other_meds:
                for med in other_meds:
                    st.markdown(f"- {med}")
            else:
                st.markdown("None recorded")

        with col2:
            # Section 4: Complaints & Symptoms
            st.markdown("""
                <div style='background-color: #FFF4E6; padding: 15px; border-radius: 8px; border-left: 4px solid #F4A261; margin-bottom: 15px;'>
                    <h4 style='color: #E76F51; margin-top: 0;'>4. Complaints & Symptoms</h4>
                </div>
            """, unsafe_allow_html=True)
            intake = patient.get('intake', {})
            complaints = intake.get('complaints_symptoms', '')
            if complaints:
                st.markdown(complaints)
            else:
                st.markdown("None recorded")
            st.markdown("")

            # Section 5: Examination Findings
            st.markdown("""
                <div style='background-color: #F0F8F0; padding: 15px; border-radius: 8px; border-left: 4px solid #52B788; margin-bottom: 15px;'>
                    <h4 style='color: #52B788; margin-top: 0;'>5. Examination Findings</h4>
                </div>
            """, unsafe_allow_html=True)
            exam_findings = intake.get('examination_findings', '')
            if exam_findings:
                st.markdown(exam_findings)
            else:
                st.markdown("None recorded")
            st.markdown("")

            # Section 6: Significant history prior to today's visit
            st.markdown("""
                <div style='background-color: #F5F0FF; padding: 15px; border-radius: 8px; border-left: 4px solid #7B68AB; margin-bottom: 15px;'>
                    <h4 style='color: #7B68AB; margin-top: 0;'>6. Significant history prior to today's visit</h4>
                </div>
            """, unsafe_allow_html=True)
            prior_history = intake.get('prior_history', '')
            if prior_history:
                st.markdown(prior_history)
            else:
                st.markdown("None recorded")

            # Section 7: Today's laboratory results
            st.markdown("""
                <div style='background-color: #FFF9E6; padding: 15px; border-radius: 8px; border-left: 4px solid #FFC107; margin-bottom: 15px;'>
                    <h4 style='color: #856404; margin-top: 0;'>7. Today's laboratory results</h4>
                </div>
            """, unsafe_allow_html=True)
            labs_raw = patient.get("labs") or {}
            encounter_date_str = patient.get("today_encounter", {}).get("date", "")
            
            # Display today's narrative lab text
            todays_labs_found = False
            if "labs_narrative" in labs_raw:
                narrative_entries = labs_raw.get("labs_narrative", [])
                for entry in narrative_entries:
                    if isinstance(entry, dict):
                        source = entry.get("source", "")
                        if source == "today":
                            narrative = entry.get("narrative_text", "")
                            if narrative:
                                st.markdown(narrative)
                                todays_labs_found = True
            
            if not todays_labs_found:
                st.markdown("None recorded")
            st.markdown("")
            
            # Section 8: Previous Laboratory results
            st.markdown("""
                <div style='background-color: #E6F3FF; padding: 15px; border-radius: 8px; border-left: 4px solid #4A90E2; margin-bottom: 15px;'>
                    <h4 style='color: #2E5C8A; margin-top: 0;'>8. Previous Laboratory results</h4>
                </div>
            """, unsafe_allow_html=True)
            
            # Display previous narrative lab text
            previous_labs_found = False
            if "labs_narrative" in labs_raw:
                narrative_entries = labs_raw.get("labs_narrative", [])
                for entry in narrative_entries:
                    if isinstance(entry, dict):
                        source = entry.get("source", "")
                        if source == "previous":
                            narrative = entry.get("narrative_text", "")
                            if narrative:
                                st.markdown(narrative)
                                st.markdown("")
                                previous_labs_found = True
            
            if not previous_labs_found:
                st.markdown("None recorded")

        # Rationale: clinicians may need to remove a custom test case and re-enter it.
        with st.expander("Delete Record", expanded=False):
            if not can_delete_selected:
                st.caption(
                    "Delete is available only for custom cases you added (stored in Data/custom_patients.json)."
                )
            if st.button(
                "Delete patient",
                key="delete_selected_patient",
                disabled=not can_delete_selected,
            ):
                st.session_state["pending_delete_patient_id"] = selected_patient_id

            if (
                st.session_state.get("pending_delete_patient_id") == selected_patient_id
                and can_delete_selected
            ):
                st.warning(
                    "This will permanently delete the selected patient case and cannot be reversed."
                )
                _confirm_key = f"confirm_delete_patient_{selected_patient_id}"
                _confirmed = st.checkbox(
                    "I understand this cannot be reversed",
                    key=_confirm_key,
                )
                if st.button(
                    "Confirm delete",
                    key=f"confirm_delete_{selected_patient_id}",
                    disabled=not _confirmed,
                ):
                    custom_patients = [
                        p
                        for p in (custom_patients or [])
                        if str(p.get("patient_id") or "") != selected_patient_id
                    ]
                    st.session_state["custom_patients"] = custom_patients
                    _save_custom_patients(CUSTOM_PATIENTS_PATH, custom_patients)

                    # Rationale: reset derived UI state that may reference the deleted patient.
                    # IMPORTANT: do not set selected_patient_label here; the selectbox widget
                    # with key selected_patient_label has already been instantiated this run.
                    # On rerun, the pre-selectbox guard will reset the selection if it no
                    # longer exists in the patient list.
                    st.session_state["analysis_ran"] = False
                    st.session_state["analysis_results"] = []
                    st.session_state["finalized"] = False
                    st.session_state.pop("pending_delete_patient_id", None)
                    st.success(f"Deleted patient case: {selected_patient_id}")
                    st.rerun()

    today_note_default = (patient.get("today_encounter", {}) or {}).get("note") or ""
    # Rationale: key by patient_id so switching patients shows the correct note input.
    today_note = st.text_area(
        "Clinician's Assessment & Plan",
        value=today_note_default,
        height=140,
        key=f"today_note_{patient.get('patient_id')}",
    )

    with st.expander("Advanced settings", expanded=False):
        prefer_chroma = st.checkbox("Prefer Chroma (persistent)", value=True)
        embedding_model_name = st.text_input(
            "Embedding model (SentenceTransformers)",
            value="all-MiniLM-L6-v2",
        )
        top_k = st.slider("Top-K guideline chunks", min_value=1, max_value=10, value=5)
        
        # Backend: index and retrieve from all pages by default (no UI controls)
        index_max_pages: Optional[int] = None
        retrieval_page_range: Optional[tuple[int, int]] = None
        
        # Agentic plan (experimental): enable ARTEMIS Review + debug expanders
        _agentic_ui_default = bool(st.session_state.get("agentic_ui_debug_enabled", False))
        agentic_ui_debug_enabled = st.checkbox(
            "Enable agentic ARTEMIS Review (experimental)",
            value=_agentic_ui_default,
            help=(
                "When enabled, runs the agentic planner using the LLM to generate a consolidated "
                "patient-level management plan (JSON) and related debug info. Requires 'Use Ollama' above."
            ),
        )
        st.session_state["agentic_ui_debug_enabled"] = bool(agentic_ui_debug_enabled)

    with st.expander("LLM settings", expanded=False):
        use_ollama = st.checkbox("Use Ollama for explanations (if available)", value=True)
        llm_mode = st.selectbox(
            "LLM mode",
            [
                "LLM screening (generate alerts)",
            ],
            index=0,
        )
        env_model = (os.getenv("OLLAMA_MODEL") or "").strip()

        discovered_models = _list_ollama_models()
        base_options: List[str] = []
        if discovered_models:
            base_options = list(dict.fromkeys(discovered_models))  # preserve order, de-dup
        else:
            base_options = ["aadide/medgemma-1.5-4b-it-Q4_K_S"]

        # Ensure the default demo model is always present as a fallback.
        if "aadide/medgemma-1.5-4b-it-Q4_K_S" not in base_options:
            base_options.append("aadide/medgemma-1.5-4b-it-Q4_K_S")

        model_options = base_options + ["Custom..."]
        default_index = 0
        if env_model and env_model in base_options:
            default_index = base_options.index(env_model)

        model_choice = st.selectbox("Ollama model", model_options, index=default_index)
        if model_choice == "Custom...":
            ollama_model_ui = st.text_input("Custom model tag", value=(env_model or ""))
        else:
            ollama_model_ui = model_choice
        if env_model:
            st.caption(f"OLLAMA_MODEL is set: {env_model} (UI model selection will be ignored)")
        num_ctx = st.number_input(
            "Context window (num_ctx)",
            min_value=1024,
            max_value=131072,
            value=8192,
            step=1024,
        )
        st.session_state["llm_num_ctx"] = int(num_ctx)

    save_disabled = not bool(GUIDELINE_PDF_PATHS)
    if st.button("Save encounter (run checks)", disabled=save_disabled):
        _run_analysis(
            patient=patient,
            today_note=today_note,
            prefer_chroma=prefer_chroma,
            embedding_model_name=embedding_model_name,
            top_k=top_k,
            index_max_pages=index_max_pages,
            retrieval_page_range=retrieval_page_range,
            use_ollama=use_ollama,
            llm_mode=llm_mode,
            ollama_model=ollama_model_ui,
            ollama_num_ctx=int(num_ctx),
            show_llm_narrative=False,
        )

    st.subheader("Results")
    if not st.session_state.get("analysis_ran", False):
        st.info("Click 'Save encounter (run checks)' to run the analysis.")
        return

    alerts: List[Alert] = st.session_state.get("analysis_alerts", [])
    status = st.session_state.get("analysis_status")

    if status == "GREEN":
        _mode = st.session_state.get("analysis_llm_mode")
        _use_ollama = st.session_state.get("analysis_use_ollama")
        _ollama_err = st.session_state.get("analysis_ollama_error")
        # Rationale: when LLM screening is configured but fails, avoid showing a
        # reassuring GREEN banner, since the screening layer is incomplete.
        llm_screening_incomplete = (
            _mode == "LLM screening (generate alerts)" and _use_ollama and bool(_ollama_err)
        )
        # Rationale: only show a GREEN banner when the full screening layer has
        # run successfully (deterministic + LLM screening as configured). For
        # runs where only deterministic rules were evaluated, use an
        # informational banner instead of GREEN to avoid over-reassurance.
        _dbg = st.session_state.get("analysis_screening_debug") or {}
        _parse_status = _dbg.get("parse_status") or "unknown"
        llm_screening_complete = (
            _mode == "LLM screening (generate alerts)"
            and _use_ollama
            and not _ollama_err
            and _parse_status in {"parsed_empty", "parsed_nonempty"}
        )
        if llm_screening_incomplete:
            st.warning(
                "INDETERMINATE: Deterministic rule-based checks did not trigger any alerts, "
                "but LLM screening failed, so additional issues may have been missed."
            )
        elif llm_screening_complete:
            st.success("GREEN: No alerts detected.")
        else:
            st.info(
                "No deterministic alerts were triggered. LLM screening was not executed "
                "for this run, so additional guideline-based issues may still exist."
            )

        _active_range = st.session_state.get("analysis_retrieval_page_range")
        if _active_range is None:
            _range_text = "all available guideline pages"
        else:
            lo, hi = _active_range
            _range_text = f"guideline pages {lo}–{hi}"

        if _mode == "LLM screening (generate alerts)" and _use_ollama and not _ollama_err:
            st.markdown(
                "No alerts were generated because the LLM screening step reviewed the Stage 1 patient "
                f"summary together with guideline excerpts from {_range_text} and did not find any "
                "guideline-based safety issues that met its alert thresholds (for example, severe anemia "
                "on zidovudine/AZT, ART toxicities, missing or overdue safety labs, or missing guideline-"
                "recommended monitoring)."
            )
            _dbg = st.session_state.get("analysis_screening_debug") or {}
            _facts = st.session_state.get("analysis_screening_patient_facts") or {}
            _parse_status = _dbg.get("parse_status") or "unknown"
            _bundle = _dbg.get("screening_bundle") or []
            _meds = (_facts.get("medications_detected") or []) if isinstance(_facts, dict) else []
            _symptoms = (_facts.get("symptoms_detected") or []) if isinstance(_facts, dict) else []
            st.caption(
                f"LLM screening audit: parse_status={_parse_status}; guideline_excerpts={len(_bundle)}; "
                f"meds_detected={len(_meds)}; symptoms_detected={len(_symptoms)}"
            )
            with st.expander("No-alert audit trail (LLM screening)", expanded=True):
                if not _dbg and not _facts:
                    st.info(
                        "Audit data was not captured for this run. Click Save again to re-run the analysis "
                        "and populate the no-alert audit trail."
                    )
                st.markdown("**What patient facts were checked (from Patient History):**")
                st.json(_facts)
                st.markdown("**What guidelines were checked (retrieved excerpts):**")
                st.json(_bundle)
                st.markdown("**How the conclusion was reached:**")
                st.markdown(
                    "- The model was instructed to extract rules ONLY from the retrieved guideline excerpts.\n"
                    "- It compared those extracted rules to the patient facts in the history and Stage 1 summary.\n"
                    f"- LLM output status: `{_dbg.get('parse_status') or 'unknown'}`."
                )
                if _dbg.get("parse_status") == "parse_failed":
                    st.error("LLM screening output could not be parsed as a JSON array; no-alert conclusion may be unreliable.")
                    st.text(_dbg.get("raw_output") or "")
                # Rationale: show raw output for all cases to debug empty output issues
                with st.expander("Debug: Raw LLM output", expanded=False):
                    st.text(_dbg.get("raw_output") or "(no output captured)")
        elif _mode == "LLM screening (generate alerts)" and _use_ollama and _ollama_err:
            st.markdown(
                "No alerts were generated. Deterministic rule-based checks did not trigger any alerts. "
                "LLM screening was configured but the Ollama call failed, so it could not contribute "
                "additional alerts."
            )
            st.caption(f"Ollama debug: {_ollama_err}")
        else:
            st.markdown(
                "No alerts were generated because the deterministic rule-based checks evaluated the "
                "structured patient context and none of the configured rules were triggered."
            )
    elif status == "RED":
        st.error(
            f"RED: {len(alerts)} alert(s) to consider (at least one high/critical issue)."
        )
    else:
        st.warning(f"YELLOW: {len(alerts)} alert(s) to consider.")
    
    # Rationale: show raw LLM output for all cases (GREEN, YELLOW, RED) to debug parsing issues
    if st.session_state.get("analysis_llm_mode") == "LLM screening (generate alerts)":
        _dbg = st.session_state.get("analysis_screening_debug") or {}
        if _dbg.get("raw_output"):
            with st.expander("Debug: Raw LLM screening output", expanded=False):
                st.caption(f"Parse status: {_dbg.get('parse_status') or 'unknown'}")
                # Rationale: show dedup stats so we can verify repetition removal is working.
                if _dbg.get("dedup_stats"):
                    st.caption("Dedup stats per alert:")
                    for ds in _dbg["dedup_stats"]:
                        st.caption(
                            f"  {ds['alert_id']}: {ds['raw_words']}→{ds['clean_words']} words, "
                            f"{ds['raw_chars']}→{ds['clean_chars']} chars"
                        )
                st.text(_dbg.get("raw_output"))

    # Rationale: use the live checkbox key ("agentic_ui_debug_enabled") so the plan
    # expanders render as soon as the checkbox is ticked, not only after a new run.
    agentic_enabled = bool(
        st.session_state.get("agentic_ui_debug_enabled", False)
        or st.session_state.get("agentic_debug_enabled", False)
    )
    # Rationale: this section is troubleshooting-only and does not help clinicians.
    # Show it only when explicitly enabled via the debug flag.
    _agentic_debug_visible = (os.getenv("AGENTIC_RAG_DEBUG") or "").strip() == "1"
    if agentic_enabled and _agentic_debug_visible:
        with st.expander("Agentic RAG debug (experimental)", expanded=False):
            agentic_result = st.session_state.get("agentic_debug_result")
            agentic_reasoner = st.session_state.get("agentic_debug_reasoner")

            if agentic_result is None and not agentic_reasoner:
                st.caption("Agentic debug is enabled but no debug data was captured for this run.")
            else:
                st.markdown("**Planner + per-subtask retrieval (Phase 1 skeleton)**")
                if isinstance(agentic_result, dict) and agentic_result.get("error"):
                    st.error(f"Agentic flow error: {agentic_result.get('error')}")
                    st.json(agentic_result)
                elif agentic_result is not None:
                    if is_dataclass(agentic_result):
                        try:
                            st.json(asdict(agentic_result))
                        except Exception:
                            st.write(agentic_result)
                    else:
                        st.write(agentic_result)

                st.markdown("**Reasoner snapshot (post-synthesis context)**")
                if agentic_reasoner is not None:
                    st.json(agentic_reasoner)
                else:
                    st.caption("No reasoner snapshot captured.")

    # Rationale: the UPDATED MANAGEMENT PLAN is rendered under the single "Output" section
    # to avoid duplicating the same content in multiple expanders.
    if False and agentic_enabled:
        agentic_result = st.session_state.get("agentic_debug_result")
        updated_plan_text = None
        try:
            if is_dataclass(agentic_result):
                updated_plan_text = getattr(agentic_result, "updated_management_plan_text", None)
                if updated_plan_text is None:
                    dbg = getattr(agentic_result, "debug_info", None) or {}
                    if isinstance(dbg, dict):
                        updated_plan_text = dbg.get("updated_management_plan_text")
        except Exception:
            updated_plan_text = None

        if isinstance(updated_plan_text, str) and updated_plan_text.strip():
            # Rationale: per clinician preference, hide the entire plan section unless we can
            # render a clean, readable plan.
            obj = None
            plan_html_text = None
            try:
                # Rationale: LLM may return raw text with code fences or preamble;
                # strip fences and extract the JSON object robustly before parsing.
                _raw = updated_plan_text.strip()
                if _raw.startswith("```"):
                    _lines = _raw.splitlines()
                    if _lines[0].strip().startswith("```"):
                        _lines = _lines[1:]
                    if _lines and _lines[-1].strip().startswith("```"):
                        _lines = _lines[:-1]
                    _raw = "\n".join(_lines).strip()
                # If there's preamble before the JSON object, find the first '{'
                _brace = _raw.find("{")
                if _brace > 0:
                    _raw = _raw[_brace:]
                # Extract the first complete top-level JSON object using brace matching
                _start = _raw.find("{")
                if _start == -1:
                    raise ValueError("No JSON object start '{' found in plan text")
                _depth = 0
                _in_str = False
                _esc = False
                _end = None
                for _i in range(_start, len(_raw)):
                    _ch = _raw[_i]
                    if _in_str:
                        if _esc:
                            _esc = False
                        elif _ch == "\\":
                            _esc = True
                        elif _ch == '"':
                            _in_str = False
                    else:
                        if _ch == '"':
                            _in_str = True
                        elif _ch == '{':
                            _depth += 1
                        elif _ch == '}':
                            _depth -= 1
                            if _depth == 0:
                                _end = _i + 1
                                break
                _slice = _raw[_start:_end] if _end is not None else _raw[_start:]
                # Fallback: tolerate minor JSON formatting issues from LLM output
                def _json_loads_relaxed(text: str):
                    try:
                        return json.loads(text)
                    except Exception:
                        t = text
                        # 1) Remove trailing commas before closing } or ]
                        t = re.sub(r',\s*([}\]])', r'\1', t)
                        # 2a) Quote single-quoted keys if present
                        t = re.sub(r"([\{\[,]\s*)'([^']+)'\s*:\s*", r'\1"\2": ', t)
                        # 2b) Quote bareword keys (letters/underscore/hyphen) if unquoted
                        t = re.sub(r'([\{\[,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)\s*:', r'\1"\2":', t)
                        # 2c) Quote known keys if unquoted (belt-and-suspenders for specific schema)
                        keys_pat = r'(art_regimen_decision|problems|monitoring_plan|patient_counselling|decision|reason|action|problem|clinician_plan_for_this_problem|explanation)'
                        t = re.sub(r'([\{,\[]\s*)' + keys_pat + r'\s*:', r'\1"\2":', t)
                        # 3) Convert single-quoted string values to double quotes (simple heuristic)
                        t = re.sub(r':\s*\'([^\']*)\'', r': "\1"', t)
                        # 4) In arrays, convert 'item' to "item"
                        t = re.sub(r'\[\s*\'([^\']*)\'', r'["\1"', t)
                        t = re.sub(r',\s*\'([^\']*)\'', r', "\1"', t)
                        return json.loads(t)
                obj = _json_loads_relaxed(_slice)

                # Clinician-friendly rendering: mirror the cards used under Output.
                def _badge(decision: str) -> str:
                    d = (decision or "").strip().lower()
                    if d == "switch":
                        return "<span style='background:#FDECEA;color:#DC3545;padding:4px 8px;border:1px solid #DC3545;border-radius:12px;font-weight:600;'>Switch</span>"
                    if d.startswith("hold"):
                        return "<span style='background:#FFF9E6;color:#856404;padding:4px 8px;border:1px solid #FFC107;border-radius:12px;font-weight:600;'>Hold temporarily</span>"
                    return "<span style='background:#F0F8F0;color:#1B5E20;padding:4px 8px;border:1px solid #52B788;border-radius:12px;font-weight:600;'>Continue</span>"

                def _safe(v: str) -> str:
                    return html.escape(str(v or "").strip())

                plan_html = []

                # 1) Regimen Decision
                ard = obj.get("art_regimen_decision") or {}
                ard_decision = str(ard.get("decision") or "").strip()
                ard_reason = str(ard.get("reason") or "").strip()
                plan_html += [
                    "<div style='background:#FFFFFF;padding:14px;border-left:4px solid #2E86AB;margin-bottom:12px;border-radius:8px;'>",
                    f"<div style='margin-bottom:6px;'>{_badge(ard_decision)} <strong style='margin-left:8px;'>ART regimen decision</strong></div>",
                    f"<div style='color:#333;'>Reason: {_safe(ard_reason)}</div>",
                    "</div>",
                ]

                # 2) Problems → Actions (Agree/Disagree/Gap/Not Addressed)
                probs = obj.get("problems") or []
                if isinstance(probs, list) and probs:
                    items = []
                    for p in probs[:12]:
                        if not isinstance(p, dict):
                            continue
                        action = _safe(p.get("action"))
                        reason = _safe(p.get("reason"))
                        cp = p.get("clinician_plan_for_this_problem") or {}
                        decision = str(cp.get("decision") or "").strip()

                        if decision in {"Gap", "Not Addressed"}:
                            prefix = "➕ ADD"
                            line = f"<strong>Consider adding:</strong> {action}"
                        elif decision == "Disagree":
                            prefix = "🔄 REVIEW"
                            line = f"<strong>Consider instead:</strong> {action}"
                        else:
                            prefix = "✅ AGREE"
                            line = f"<strong>Agree:</strong> {action}"

                        items.append(
                            "<li style='margin-bottom:6px;'>"
                            f"<span style='margin-right:6px;'>{prefix}</span> {line}"
                            f"<div style='color:#555;font-size:12px;margin-top:2px;'>Reason: {reason}</div>"
                            "</li>"
                        )

                    if items:
                        plan_html += [
                            "<div style='background-color:#F0F8F0;padding:14px;border-left:4px solid #52B788;border-radius:8px;margin-bottom:12px;'>",
                            "<h4 style='color:#1B5E20;margin:0 0 8px 0;'>Recommended Actions</h4>",
                            "<ol style='margin:0;padding-left:18px;'>" + "".join(items) + "</ol>",
                            "</div>",
                        ]

                # 3) Monitoring Plan
                mp = obj.get("monitoring_plan")
                mp_items = []
                if isinstance(mp, list):
                    mp_items = [f"<li>{_safe(x)}</li>" for x in mp if str(x).strip()]
                elif isinstance(mp, dict):
                    for k, v in mp.items():
                        if isinstance(v, (list, tuple)):
                            for x in v:
                                if str(x).strip():
                                    mp_items.append(f"<li>{_safe(x)}</li>")
                        elif str(v).strip():
                            mp_items.append(f"<li>{_safe(v)}</li>")
                elif isinstance(mp, str) and mp.strip():
                    mp_items = [f"<li>{_safe(mp)}</li>"]

                if mp_items:
                    plan_html += [
                        "<div style='background:#E8F4F8;padding:14px;border-left:4px solid #2E86AB;border-radius:8px;margin-bottom:12px;'>",
                        "<h4 style='color:#2E86AB;margin:0 0 8px 0;'>Monitoring</h4>",
                        "<ul style='margin:0;padding-left:18px;'>" + "".join(mp_items) + "</ul>",
                        "</div>",
                    ]

                # 4) Patient Counselling
                pc = obj.get("patient_counselling") or []
                if isinstance(pc, list) and pc:
                    pc_items = [f"<li>{_safe(x)}</li>" for x in pc if str(x).strip()]
                    if pc_items:
                        plan_html += [
                            "<div style='background:#F5F0FF;padding:14px;border-left:4px solid #7B68AB;border-radius:8px;margin-bottom:12px;'>",
                            "<h4 style='color:#7B68AB;margin:0 0 8px 0;'>Counselling</h4>",
                            "<ul style='margin:0;padding-left:18px;'>" + "".join(pc_items) + "</ul>",
                            "</div>",
                        ]

                # 5) References (optional)
                refs = obj.get("references") or obj.get("citations")
                if isinstance(refs, list):
                    ref_items = []
                    for r in refs:
                        if isinstance(r, dict):
                            src = _safe(r.get("source") or r.get("src") or r.get("source_path") or "Guidelines")
                            page = r.get("page") or r.get("page_number")
                            try:
                                page_i = int(page) if page is not None else None
                            except Exception:
                                page_i = None
                            if page_i is not None:
                                ref_items.append(f"<li>{src}, p.{page_i}</li>")
                            else:
                                title = _safe(r.get("title") or "")
                                text = f"{src}{(': ' + title) if title else ''}"
                                ref_items.append(f"<li>{text}</li>")
                        else:
                            s = _safe(r)
                            if s:
                                ref_items.append(f"<li>{s}</li>")

                    if ref_items:
                        plan_html += [
                            "<div style='background:#FAFAFA;padding:14px;border-left:4px solid #9E9E9E;border-radius:8px;margin-bottom:12px;'>",
                            "<h4 style='color:#555;margin:0 0 8px 0;'>References</h4>",
                            "<ol style='margin:0;padding-left:18px;'>" + "".join(ref_items) + "</ol>",
                            "</div>",
                        ]

                plan_html_text = "\n".join(plan_html) if plan_html else None
            except Exception:
                obj = None
                plan_html_text = None

            if isinstance(obj, dict) and isinstance(plan_html_text, str) and plan_html_text.strip():
                # Rationale: match clinician-facing naming used in the example output.
                with st.expander("ARTEMIS Review.", expanded=True):
                    st.markdown(plan_html_text, unsafe_allow_html=True)

                    # Keep raw JSON available for reviewers behind a collapsed toggle.
                    with st.expander("View raw plan JSON", expanded=False):
                        st.json(obj)

    # Rationale: render per-alert expanders so clinicians can see each alert's
    # finding, explanation, and references.
    if alerts:
        for item in st.session_state.get("analysis_results", []):
            alert: Alert = item["alert"]
            retrieved: List[VectorSearchResult] = item["retrieved"]
            explanation: ExplanationResult = item["explanation"]

            with st.expander(f"Alert: {alert.title}", expanded=True):
                # Rationale: present clinician-facing text without inline citation fragments
                # and keep numbered references in one place for readability.
                _finding_text = str(alert.message or "")
                _finding_text = re.sub(
                    r"\[\s*chunk_id\s*:\s*[^\],]+\s*,\s*page_number\s*:\s*(\d+)\s*\]",
                    "",
                    _finding_text,
                    flags=re.IGNORECASE,
                )
                _finding_text = re.sub(r"\(page\s*=\s*\d+\)", "", _finding_text, flags=re.IGNORECASE)
                _finding_text = re.sub(
                    r"chunk[_\s]*id\s*[:=]\s*[^\s,\)\]]+",
                    "",
                    _finding_text,
                    flags=re.IGNORECASE,
                )

                _why_text = str(explanation.text or "")
                _why_text = re.sub(
                    r"\[\s*chunk_id\s*:\s*[^\],]+\s*,\s*page_number\s*:\s*(\d+)\s*\]",
                    "",
                    _why_text,
                    flags=re.IGNORECASE,
                )
                _why_text = re.sub(r"\(page\s*=\s*\d+\)", "", _why_text, flags=re.IGNORECASE)
                _why_text = re.sub(
                    r"chunk[_\s]*id\s*[:=]\s*[^\s,\)\]]+",
                    "",
                    _why_text,
                    flags=re.IGNORECASE,
                )

                # Rationale: some explanations include duplicated headings and large raw
                # dictionaries; strip them so the main UI reads like a normal note.
                _why_text = re.sub(r"^\s*why this alert\s*:\s*", "", _why_text, flags=re.IGNORECASE)
                _why_text = re.split(r"\bpatient evidence\s*:\s*", _why_text, maxsplit=1, flags=re.IGNORECASE)[0]
                _why_text = re.split(r"\bguideline excerpts\s*:\s*", _why_text, maxsplit=1, flags=re.IGNORECASE)[0]

                # Rationale: build a single numbered reference list from evidence + retrieved pages.
                _pages: List[int] = []
                try:
                    _cit = (alert.evidence or {}).get("citations") or []
                    if isinstance(_cit, list):
                        for c in _cit:
                            if isinstance(c, dict) and c.get("page_number") is not None:
                                try:
                                    _pages.append(int(c.get("page_number")))
                                except Exception:
                                    continue
                except Exception:
                    _pages = []
                if retrieved:
                    for r in retrieved:
                        try:
                            p = (getattr(r, "metadata", {}) or {}).get("page_number")
                            if p is not None:
                                _pages.append(int(p))
                        except Exception:
                            continue

                # Rationale: keep compatible with older Python versions (avoid built-in generics like set[int]).
                _seen_pages = set()
                _ref_pages: List[int] = []
                for p in _pages:
                    if isinstance(p, int) and p not in _seen_pages:
                        _seen_pages.add(p)
                        _ref_pages.append(p)

                st.markdown("**Finding**")
                st.write(_finding_text.strip() or "-")

                # Rationale: for LLM screening alerts, show the recommended_action
                # instead of duplicating the finding text under "Why this alert?".
                _is_checklist = (alert.evidence or {}).get("type") == "llm_audit_checklist"
                _rec_action = (alert.evidence or {}).get("recommended_action") or ""
                if _rec_action.strip():
                    st.markdown("**Recommended Action**")
                    st.write(_rec_action.strip())
                else:
                    st.markdown("**Why this alert?**")
                    st.write(_why_text.strip() or "-")

                # Rationale: keep technical retrieval details available without cluttering the main view.
                with st.expander("Details (debug)", expanded=False):
                    # Rationale: keep references out of the main alert view; only show them in Details.
                    if _ref_pages:
                        st.markdown("**References**")
                        st.markdown(
                            "\n".join([f"{i+1}. Guidelines, p.{p}" for i, p in enumerate(_ref_pages)])
                        )

                    # Rationale: move generation status/latency out of the main clinician view.
                    if _is_checklist:
                        st.caption(
                            "Checklist item generated by LLM (single-call mode). Per‑alert explanation is disabled in this mode."
                        )
                    elif explanation.used_llm:
                        st.caption("Generated with local LLM (Ollama) using retrieved excerpts.")
                        _elapsed = st.session_state.get(f"explain_time_{alert.alert_id}")
                        if isinstance(_elapsed, (int, float)):
                            st.caption(f"LLM time: {_elapsed:.2f}s")
                    else:
                        if st.session_state.get("analysis_use_ollama"):
                            st.caption("Deterministic fallback explanation (LLM unavailable).")
                            model = st.session_state.get("analysis_ollama_model")
                            err = st.session_state.get("analysis_ollama_error")
                            if model:
                                st.caption(f"Ollama model: {model}")
                            if err:
                                st.caption(f"Ollama debug: {err}")
                        else:
                            st.caption("Deterministic fallback explanation (LLM disabled).")

                    st.markdown("**Guideline retrieval**")
                    if retrieved:
                        for r in retrieved:
                            st.markdown(
                                f"- page={r.metadata.get('page_number')} | distance={r.distance:.4f} | chunk_id={r.chunk_id}"
                            )
                    else:
                        st.write("No guideline chunks retrieved.")

                # Rationale: allow promoting useful LLM-screening issues into a
                # candidate list for future deterministic rules without changing
                # rule logic at runtime.
                if (alert.evidence or {}).get("type") == "llm_screening":
                    if st.button(
                        "Mark as candidate deterministic rule",
                        key=f"candidate_rule_{alert.alert_id}",
                    ):
                        existing_rules = _load_candidate_rules(CANDIDATE_RULES_PATH)
                        # Avoid duplicating the same alert_id in the candidate file.
                        existing_ids = {
                            str(r.get("alert_id")) for r in existing_rules if isinstance(r, dict)
                        }
                        if alert.alert_id not in existing_ids:
                            candidate: Dict[str, Any] = {
                                "alert_id": alert.alert_id,
                                "title": alert.title,
                                "message": alert.message,
                                "evidence": alert.evidence,
                            }
                            existing_rules.append(candidate)
                            _save_candidate_rules(CANDIDATE_RULES_PATH, existing_rules)
                            st.success("Marked as candidate deterministic rule (saved under Data/candidate_rules.json).")

    # Trial Result - Experimental formatted output display
    # Rationale: keep a single consolidated "Output" section even when there are no alerts,
    # so the UPDATED MANAGEMENT PLAN can still be shown under Output.
    if st.session_state.get("analysis_ran"):
        with st.expander("Output", expanded=False):
            st.caption("Experimental display format for alert output")

            if agentic_enabled:
                agentic_result = st.session_state.get("agentic_debug_result")
                updated_plan_text = None
                try:
                    if is_dataclass(agentic_result):
                        updated_plan_text = getattr(agentic_result, "updated_management_plan_text", None)
                        if updated_plan_text is None:
                            dbg = getattr(agentic_result, "debug_info", None) or {}
                            if isinstance(dbg, dict):
                                updated_plan_text = dbg.get("updated_management_plan_text")
                except Exception:
                    updated_plan_text = None

                if isinstance(updated_plan_text, str) and updated_plan_text.strip():
                    # Rationale: per clinician preference, hide the plan section unless we can
                    # render a clean, readable plan.
                    obj = None
                    plan_html_text = None
                    try:
                        # Rationale: same robust JSON extraction as the standalone expander.
                        _raw2 = updated_plan_text.strip()
                        if _raw2.startswith("```"):
                            _lines2 = _raw2.splitlines()
                            if _lines2[0].strip().startswith("```"):
                                _lines2 = _lines2[1:]
                            if _lines2 and _lines2[-1].strip().startswith("```"):
                                _lines2 = _lines2[:-1]
                            _raw2 = "\n".join(_lines2).strip()
                        _brace2 = _raw2.find("{")
                        if _brace2 > 0:
                            _raw2 = _raw2[_brace2:]
                        # Extract the first complete top-level JSON object using brace matching
                        _start2 = _raw2.find("{")
                        if _start2 == -1:
                            raise ValueError("No JSON object start '{' found in plan text")
                        _depth2 = 0
                        _in_str2 = False
                        _esc2 = False
                        _end2 = None
                        for _j in range(_start2, len(_raw2)):
                            _ch2 = _raw2[_j]
                            if _in_str2:
                                if _esc2:
                                    _esc2 = False
                                elif _ch2 == "\\":
                                    _esc2 = True
                                elif _ch2 == '"':
                                    _in_str2 = False
                            else:
                                if _ch2 == '"':
                                    _in_str2 = True
                                elif _ch2 == '{':
                                    _depth2 += 1
                                elif _ch2 == '}':
                                    _depth2 -= 1
                                    if _depth2 == 0:
                                        _end2 = _j + 1
                                        break
                        _slice2 = _raw2[_start2:_end2] if _end2 is not None else _raw2[_start2:]
                        # Fallback: tolerate minor JSON formatting issues from LLM output
                        def _json_loads_relaxed2(text: str):
                            try:
                                return json.loads(text)
                            except Exception:
                                t2 = text
                                # 1) Remove trailing commas before closing } or ]
                                t2 = re.sub(r',\s*([}\]])', r'\1', t2)
                                # 2a) Quote single-quoted keys if present
                                t2 = re.sub(r"([\{\[,]\s*)'([^']+)'\s*:\s*", r'\1"\2": ', t2)
                                # 2b) Quote bareword keys (letters/underscore/hyphen) if unquoted
                                t2 = re.sub(r'([\{\[,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)\s*:', r'\1"\2":', t2)
                                # 2c) Quote known keys if unquoted
                                keys_pat2 = r'(art_regimen_decision|problems|monitoring_plan|patient_counselling|decision|reason|action|problem|clinician_plan_for_this_problem|explanation)'
                                t2 = re.sub(r'([\{,\[]\s*)' + keys_pat2 + r'\s*:', r'\1"\2":', t2)
                                # 3) Convert single-quoted string values to double quotes (simple heuristic)
                                t2 = re.sub(r':\s*\'([^\']*)\'', r': "\1"', t2)
                                # 4) In arrays, convert 'item' to "item"
                                t2 = re.sub(r'\[\s*\'([^\']*)\'', r'["\1"', t2)
                                t2 = re.sub(r',\s*\'([^\']*)\'', r', "\1"', t2)
                                return json.loads(t2)
                        obj = _json_loads_relaxed2(_slice2)

                        # Render a clinician-friendly view (no raw JSON).
                        def _badge(decision: str) -> str:
                            d = (decision or "").strip().lower()
                            if d == "switch":
                                return "<span style='background:#FDECEA;color:#DC3545;padding:4px 8px;border:1px solid #DC3545;border-radius:12px;font-weight:600;'>Switch</span>"
                            if d.startswith("hold"):
                                return "<span style='background:#FFF9E6;color:#856404;padding:4px 8px;border:1px solid #FFC107;border-radius:12px;font-weight:600;'>Hold temporarily</span>"
                            return "<span style='background:#F0F8F0;color:#1B5E20;padding:4px 8px;border:1px solid #52B788;border-radius:12px;font-weight:600;'>Continue</span>"

                        def _safe(v: str) -> str:
                            return html.escape(str(v or "").strip())

                        # 1) Regimen Decision
                        ard = obj.get("art_regimen_decision") or {}
                        ard_decision = str(ard.get("decision") or "").strip()
                        ard_reason = str(ard.get("reason") or "").strip()
                        plan_html = [
                            "<div style='background:#FFFFFF;padding:14px;border-left:4px solid #2E86AB;margin-bottom:12px;border-radius:8px;'>",
                            f"<div style='margin-bottom:6px;'>{_badge(ard_decision)} <strong style='margin-left:8px;'>ART regimen decision</strong></div>",
                            f"<div style='color:#333;'>Reason: {_safe(ard_reason)}</div>",
                            "</div>",
                        ]

                        # 2) Problems → Actions (Agree/Disagree/Gap/Not Addressed)
                        probs = obj.get("problems") or []
                        if isinstance(probs, list) and probs:
                            items = []
                            for p in probs[:12]:
                                if not isinstance(p, dict):
                                    continue
                                action = _safe(p.get("action"))
                                reason = _safe(p.get("reason"))
                                cp = p.get("clinician_plan_for_this_problem") or {}
                                decision = str(cp.get("decision") or "").strip()

                                if decision in {"Gap", "Not Addressed"}:
                                    prefix = "➕ ADD"
                                    status_label = "Not in your plan"
                                    line = f"<strong>Consider adding:</strong> {action}"
                                elif decision == "Disagree":
                                    prefix = "🔄 REVIEW"
                                    status_label = "Needs review"
                                    line = f"<strong>Consider instead:</strong> {action}"
                                else:
                                    prefix = "✅ AGREE"
                                    status_label = "Matches your plan"
                                    line = f"<strong>Agree:</strong> {action}"

                                items.append(
                                    "<li style='margin-bottom:8px;'>"
                                    f"<div><span style='margin-right:6px;'>{prefix}</span> {line}</div>"
                                    f"<div style='color:#444;font-size:12px;margin-top:2px;'><em>{_safe(status_label)}</em></div>"
                                    f"<div style='color:#555;font-size:12px;margin-top:2px;'>Reason: {reason}</div>"
                                    "</li>"
                                )

                            if items:
                                plan_html += [
                                    "<div style='background-color:#F0F8F0;padding:14px;border-left:4px solid #52B788;border-radius:8px;margin-bottom:12px;'>",
                                    "<h4 style='color:#1B5E20;margin:0 0 8px 0;'>Suggested Management Plan</h4>",
                                    "<ol style='margin:0;padding-left:18px;'>" + "".join(items) + "</ol>",
                                    "</div>",
                                ]

                        # 3) Monitoring Plan
                        mp = obj.get("monitoring_plan")
                        mp_items = []
                        if isinstance(mp, list):
                            mp_items = [f"<li>{_safe(x)}</li>" for x in mp if str(x).strip()]
                        elif isinstance(mp, dict):
                            for k, v in mp.items():
                                if isinstance(v, (list, tuple)):
                                    for x in v:
                                        if str(x).strip():
                                            mp_items.append(f"<li>{_safe(x)}</li>")
                                elif str(v).strip():
                                    mp_items.append(f"<li>{_safe(v)}</li>")
                        elif isinstance(mp, str) and mp.strip():
                            mp_items = [f"<li>{_safe(mp)}</li>"]

                        if mp_items:
                            plan_html += [
                                "<div style='background:#E8F4F8;padding:14px;border-left:4px solid #2E86AB;border-radius:8px;margin-bottom:12px;'>",
                                "<h4 style='color:#2E86AB;margin:0 0 8px 0;'>Monitoring</h4>",
                                "<ul style='margin:0;padding-left:18px;'>" + "".join(mp_items) + "</ul>",
                                "</div>",
                            ]

                        # 4) Patient Counselling
                        pc = obj.get("patient_counselling") or []
                        if isinstance(pc, list) and pc:
                            pc_items = [f"<li>{_safe(x)}</li>" for x in pc if str(x).strip()]
                            if pc_items:
                                plan_html += [
                                    "<div style='background:#F5F0FF;padding:14px;border-left:4px solid #7B68AB;border-radius:8px;margin-bottom:12px;'>",
                                    "<h4 style='color:#7B68AB;margin:0 0 8px 0;'>Counselling</h4>",
                                    "<ul style='margin:0;padding-left:18px;'>" + "".join(pc_items) + "</ul>",
                                    "</div>",
                                ]

                        plan_html_text = "\n".join(plan_html) if plan_html else None
                    except Exception:
                        obj = None
                        plan_html_text = None

                    if isinstance(obj, dict) and isinstance(plan_html_text, str) and plan_html_text.strip():
                        # Rationale: match clinician-facing naming used in the example output.
                        st.markdown("### ARTEMIS Review.")
                        st.markdown(plan_html_text, unsafe_allow_html=True)

                        # Optional: raw JSON for reviewers (collapsed UI)
                        with st.expander("View raw plan JSON", expanded=False):
                            st.json(obj)


    # Rationale: keep review controls near the Finalize button
    if alerts and st.session_state.get("analysis_ran"):
        analysis_results = st.session_state.get("analysis_results") or []
        if analysis_results:
            selected_alert_id = st.session_state.get("output_selected_alert_id")
            if not selected_alert_id:
                selected_alert_id = analysis_results[0]["alert"].alert_id
                st.session_state["output_selected_alert_id"] = selected_alert_id
            if selected_alert_id not in {r["alert"].alert_id for r in analysis_results}:
                selected_alert_id = analysis_results[0]["alert"].alert_id
                st.session_state["output_selected_alert_id"] = selected_alert_id

            selected_result = next(
                (r for r in analysis_results if r["alert"].alert_id == selected_alert_id),
                analysis_results[0],
            )
            alert = selected_result["alert"]

            st.markdown("**Acknowledge / Override**")
            st.radio(
                "Action",
                ["Unreviewed", "Acknowledge", "Override"],
                key=f"alert_action_{alert.alert_id}",
                horizontal=True,
            )

            if st.session_state.get(f"alert_action_{alert.alert_id}") == "Override":
                st.selectbox(
                    "Override reason (required)",
                    [
                        "",
                        "Already addressed",
                        "Not applicable",
                        "Will address later",
                        "Patient declined",
                        "Other",
                    ],
                    key=f"alert_override_reason_{alert.alert_id}",
                )
                st.text_area(
                    "Override comment (optional)",
                    key=f"alert_override_comment_{alert.alert_id}",
                    height=80,
                )

    can_finalize = _can_finalize(alerts)
    if st.button("Finalize / Close visit", disabled=not can_finalize):
        st.session_state["finalized"] = True

    if st.session_state.get("finalized"):
        st.success("Visit finalized.")
    elif st.session_state.get("analysis_ran") and alerts and not can_finalize:
        st.info("Finalize is disabled until all alerts are acknowledged or overridden with a reason.")

    st.markdown("""
        <div style='background-color: #E8F4F8; padding: 15px; border-radius: 8px; border-left: 4px solid #2E86AB; margin: 20px 0;'>
            <h3 style='color: #2E86AB; margin-top: 0;'>Status</h3>
            <p style='margin-bottom: 0;'>Workflow active: Save runs checks; Finalize requires review.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Guidelines section moved to bottom
    with st.expander("📚 Clinical Guidelines Reference", expanded=False):
        st.markdown("""
            <div style='background-color: #F5F0FF; padding: 15px; border-radius: 8px; border-left: 4px solid #7B68AB;'>
                <h4 style='color: #7B68AB; margin-top: 0;'>Guidelines Used</h4>
            </div>
        """, unsafe_allow_html=True)
        if GUIDELINE_PATHS:
            st.write("Using Markdown guideline files:")
            for p in GUIDELINE_PATHS:
                st.markdown(f"- `{p.name}`")
            st.success(f"{len(GUIDELINE_PATHS)} guideline file(s) found.")
        else:
            st.error(
                "Guidelines PDF not found. Place it under `Data/` as specified in `initial-prompt-template.md`."
            )


if __name__ == "__main__":
    main()
