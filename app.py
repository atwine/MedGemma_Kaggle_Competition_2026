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

    st.title("HIV Clinical Nudge Engine (Demo)")
    st.caption(
        "Decision support only. Clinician retains final authority. Synthetic data only."
    )

    st.subheader("Guidelines")
    if GUIDELINE_PATHS:
        st.write("Using Markdown guideline files:")
        for p in GUIDELINE_PATHS:
            st.write(f"- `{p.name}`")
        st.success(f"{len(GUIDELINE_PATHS)} guideline file(s) found.")
    else:
        st.error(
            "Guidelines PDF not found. Place it under `Data/` as specified in `initial-prompt-template.md`."
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
        st.warning(
            "For testing: avoid entering real names/identifiers. Use coded IDs or de-identified data."
        )

        new_patient_id = st.text_input("Patient ID", key="new_case_patient_id")

        st.caption("Optional: enter patient history, including any prior labs and events.")
        new_prev_note = st.text_area("Patient History", key="new_case_prev_note", height=80)
        history_text = (new_prev_note or "").strip()
        approx_tokens = len(history_text.split()) if history_text else 0
        current_ctx = st.session_state.get("llm_num_ctx")
        if current_ctx:
            st.caption(f"Approx. history length: ~{approx_tokens} tokens vs context window {current_ctx} tokens.")
        else:
            st.caption(f"Approx. history length: ~{approx_tokens} tokens.")

        if st.button("Add case to patient list"):
            pid = (new_patient_id or "").strip()
            if not pid:
                st.error("Patient ID is required.")
            elif any((p.get("patient_id") == pid) for p in (list(patients) + custom_patients)):
                st.error("That Patient ID already exists in the current patient list.")
            else:
                encounter_iso = datetime.date.today().isoformat()

                patient_record: Dict[str, Any] = {
                    "patient_id": pid,
                    "name": pid,
                    "art_regimen_current": [],
                    "visits": [],
                    "labs": {},
                    "today_encounter": {
                        "date": encounter_iso,
                        # Rationale: the add-case form captures history only; the
                        # clinician enters the current-day note after selecting the
                        # patient from the list.
                        "note": "",
                        "orders": [],
                        "med_changes": [],
                    },
                }

                prev_note = (new_prev_note or "").strip()
                if prev_note:
                    patient_record["visits"].append(
                        {
                            "date": encounter_iso,
                            "type": "routine",
                            "clinician_note": prev_note,
                        }
                    )

                # Rationale: keep the local list in sync so the newly added patient
                # appears in the list immediately on this rerun.
                custom_patients = custom_patients + [patient_record]
                st.session_state["custom_patients"] = custom_patients
                _save_custom_patients(CUSTOM_PATIENTS_PATH, custom_patients)
                # Rationale: auto-select the newly added patient to make it clear
                # the add succeeded.
                st.session_state["selected_patient_label"] = f"{pid} - {patient_record['name']}"
                st.success("Case added. Select it from the patient list below.")

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

    today_note_default = (patient.get("today_encounter", {}) or {}).get("note") or ""
    # Rationale: key by patient_id so switching patients shows the correct note input.
    today_note = st.text_area(
        "Today's encounter note",
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
        top_k = st.slider("Top-K guideline chunks", min_value=1, max_value=10, value=10)
        # Rationale: default to indexing all pages so Markdown tables/flowcharts are
        # consistently embedded across the entire document set.
        index_max_pages: Optional[int] = None
        index_pages_choice = st.selectbox(
            "Index max pages",
            options=["All"],
            index=0,
        )
        index_max_pages = None

        # Rationale: Stage 2 — optional retrieval filter by guideline page range.
        retrieval_page_range: Optional[tuple[int, int]] = None
        retrieval_pages_choice = st.selectbox(
            "Retrieval page range",
            # Rationale: default to querying across all pages so no guideline
            # content is unintentionally excluded.
            options=["All"],
            index=0,
            help="Constrain RAG retrieval to specific guideline pages (uses chunk metadata).",
        )
        retrieval_page_range = None

        st.checkbox(
            "Enable Agentic RAG debug (experimental)",
            key="agentic_ui_debug_enabled",
            value=True,
        )

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
            value=32000,
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

    # UX: Show the active retrieval page range to make bounds explicit for reviewers.
    _active_range = st.session_state.get("analysis_retrieval_page_range")
    if _active_range is None:
        st.caption("Retrieval page range: All pages")
    else:
        lo, hi = _active_range
        st.caption(f"Retrieval page range: pages {lo}–{hi}")

    agentic_enabled = st.session_state.get("agentic_debug_enabled", False)
    if agentic_enabled:
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

    # Rationale: surface the new Part 3 "UPDATED MANAGEMENT PLAN" output in the
    # main Results flow (before alert review actions) so it is easy to find.
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
            with st.expander("UPDATED MANAGEMENT PLAN (NEW)", expanded=True):
                try:
                    obj = json.loads(updated_plan_text)
                    st.json(obj)
                except Exception:
                    st.text(updated_plan_text)

    if alerts:
        for item in st.session_state.get("analysis_results", []):
            alert: Alert = item["alert"]
            retrieved: List[VectorSearchResult] = item["retrieved"]
            explanation: ExplanationResult = item["explanation"]

            with st.expander(f"Alert: {alert.title}", expanded=True):
                st.write(alert.message)
                st.json(alert.evidence)

                st.markdown("**Guideline retrieval**")
                if retrieved:
                    for r in retrieved:
                        st.markdown(
                            f"- page={r.metadata.get('page_number')} | distance={r.distance:.4f} | chunk_id={r.chunk_id}"
                        )
                else:
                    st.write("No guideline chunks retrieved.")

                st.markdown("**Why this alert?**")
                _is_checklist = (alert.evidence or {}).get("type") == "llm_audit_checklist"
                if _is_checklist:
                    # UX: checklist items are generated via a single LLM call; per‑alert explanations are skipped.
                    st.caption("Checklist item generated by LLM (single-call mode). Per‑alert explanation is disabled in this mode.")
                elif explanation.used_llm:
                    st.caption("Generated with local LLM (Ollama) using retrieved excerpts.")
                    # Rationale: show latency for per‑alert explanations when LLM was used.
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
                st.write(explanation.text)

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

    st.subheader("Status")
    st.info("Workflow active: Save runs checks; Finalize requires review.")

    with st.expander("Candidate deterministic rules (from LLM screening)", expanded=False):
        rules = _load_candidate_rules(CANDIDATE_RULES_PATH)
        if not rules:
            st.caption(
                "No candidate rules have been marked yet. Use 'Mark as candidate deterministic rule' "
                "on an LLM-screening alert to add one."
            )
        else:
            st.json(rules)


if __name__ == "__main__":
    main()
