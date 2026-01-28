"""Streamlit demo entrypoint for the HIV Clinical Nudge Engine.
 
 This app implements an initial local-first “Save vs Finalize” workflow:
 - Save runs deterministic alert rules + RAG retrieval (and optional Ollama explanation)
 - Finalize is blocked until each alert is acknowledged or overridden with a reason
 """
 
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from modules.alert_rules import Alert, run_alerts
from modules.embedder import Embedder, EmbedderConfig
from modules.explanation_generator import (
    ExplanationResult,
    generate_audit_checklist_alerts,
    generate_explanation,
)
from modules.llm_client import OllamaClient, OllamaConfig
from modules.patient_parser import build_patient_context, load_mock_patients
from modules.rag_engine import RagEngine
from modules.vector_store import VectorSearchResult, create_vector_store


PROJECT_ROOT = Path(__file__).resolve().parent
CONSOLIDATED_GUIDELINE_PDF_PATH = PROJECT_ROOT / "Data" / "Consolidated-HIV-and-AIDS-Guidelines-20230516.pdf"
LEGACY_GUIDELINE_PDF_PATH = PROJECT_ROOT / "Data" / "Uganda Clinical Guidelines 2023.pdf"
NCD_GUIDELINE_PDF_PATH = (
    PROJECT_ROOT
    / "Data"
    / "Uganda+Integrated+Guidelines+for+the+management+of+NCDs-+Uganda+with+cover-FIN+14October2019.V4.pdf"
)

# Rationale: prefer the newer consolidated guideline PDF when present, but keep a
# fallback to the legacy filename to avoid breaking existing setups.
# Note: for this demo we embed/index only the consolidated guideline PDF.
GUIDELINE_PDF_PATHS = [p for p in [CONSOLIDATED_GUIDELINE_PDF_PATH] if p.exists()]
MOCK_PATIENTS_PATH = PROJECT_ROOT / "Data" / "mock_patients.json"


@st.cache_data
def _load_patients(path: str) -> List[Dict[str, Any]]:
    # Rationale: cache synthetic demo data across reruns for responsiveness.
    return load_mock_patients(Path(path))
 
 
@st.cache_resource
def _get_embedder(model_name: str) -> Embedder:
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
    embedder = _get_embedder(embedding_model_name)
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
    # active guideline PDF in the collection name to prevent mixing chunks from
    # previous documents and to force a clean re-index when the source PDF changes.
    guideline_key = "no_guideline"
    if GUIDELINE_PDF_PATHS:
        guideline_key = (
            GUIDELINE_PDF_PATHS[0].stem.strip().lower().replace(" ", "_").replace("+", "_")
        )
    vector_store = create_vector_store(
        project_root=PROJECT_ROOT,
        prefer_chroma=prefer_chroma,
        collection_name=f"uganda_hiv_guidelines__{model_key}__{guideline_key}",
    )
    return RagEngine(
        project_root=PROJECT_ROOT,
        guideline_pdf_paths=GUIDELINE_PDF_PATHS,
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
    # Rationale: allow adding ad-hoc test cases without editing JSON files.
    st.session_state.setdefault("custom_patients", [])


def _compute_overall_status(alerts: List[Alert]) -> str:
    return "YELLOW" if alerts else "GREEN"


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
) -> None:
    # Rationale: keep analysis deterministic for alert triggering, and only use the
    # LLM for explanation/citation if available.

    patient_copy = copy.deepcopy(patient)
    patient_copy.setdefault("today_encounter", {})
    patient_copy["today_encounter"]["note"] = today_note

    context = build_patient_context(patient_copy)
    alerts = run_alerts(context)

    rag = _get_rag_engine(
        prefer_chroma=prefer_chroma,
        embedding_model_name=embedding_model_name,
        # Rationale: cache-bust to refresh class definition after code change (Stage 2 API).
        engine_version=3,
    )

    # Rationale: indexing can be expensive; we allow limiting pages for a faster demo.
    if GUIDELINE_PDF_PATHS:
        rag.ensure_indexed(max_pages=index_max_pages)

    llm_client = None
    if use_ollama:
        cfg = OllamaConfig(model=(ollama_model or "aadide/medgemma-1.5-4b-it-Q4_K_S"), num_ctx=ollama_num_ctx)
        llm_client = OllamaClient(cfg)

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
    # Manage LLM calls by mode: checklist vs per‑alert explanations.
    checklist_llm = llm_client if (use_ollama and llm_mode in ["Checklist only", "Synthesis"]) else None
    explanation_llm = llm_client if (use_ollama and llm_mode == "Per-alert explanations") else None

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

    # Rationale: reset per-alert review state for each analysis run. Without this,
    # a previous acknowledgment/override can incorrectly allow finalize on a new
    # Save without review.
    for alert in alerts:
        st.session_state[f"alert_action_{alert.alert_id}"] = "Unreviewed"
        st.session_state[f"alert_override_reason_{alert.alert_id}"] = ""
        st.session_state[f"alert_override_comment_{alert.alert_id}"] = ""

    results: List[Dict[str, Any]] = []
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

            if ollama_status is not None and explanation_llm is not None:
                # Rationale: per-alert LLM calls may each take time; show which step is active.
                ollama_status.update(
                    label=f"Ollama: generating explanation for '{alert.title}'…",
                    state="running",
                )
            explanation = generate_explanation(
                patient_context=context,
                alert=alert,
                retrieved_chunks=retrieved,
                llm_client=explanation_llm,
            )

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

    st.session_state["analysis_ran"] = True
    st.session_state["finalized"] = False
    st.session_state["analysis_alerts"] = alerts
    st.session_state["analysis_results"] = results
    st.session_state["analysis_status"] = _compute_overall_status(alerts)
    st.session_state["analysis_use_ollama"] = bool(use_ollama)
    _client_used = (checklist_llm or explanation_llm)
    st.session_state["analysis_ollama_model"] = _client_used.model if _client_used is not None else None
    st.session_state["analysis_ollama_error"] = ollama_error


def main() -> None:
    st.set_page_config(page_title="HIV Clinical Nudge Engine", layout="wide")
    _init_session_state()

    st.title("HIV Clinical Nudge Engine (Demo)")
    st.caption(
        "Decision support only. Clinician retains final authority. Synthetic data only."
    )

    st.subheader("Guidelines")
    if GUIDELINE_PDF_PATHS:
        st.write("Using local PDFs:")
        for p in GUIDELINE_PDF_PATHS:
            st.write(f"- `{p}`")
        st.success("Guidelines PDF found.")
    else:
        st.error(
            "Guidelines PDF not found. Place it under `Data/` as specified in `initial-prompt-template.md`."
        )

    if not MOCK_PATIENTS_PATH.exists():
        st.error("Mock patients file not found: `Data/mock_patients.json`")
        return

    patients = _load_patients(str(MOCK_PATIENTS_PATH))
    if not patients:
        st.error("No mock patients found in `Data/mock_patients.json`")
        return

    custom_patients = list(st.session_state.get("custom_patients") or [])

    with st.expander("Add patient case", expanded=False):
        st.warning(
            "For testing: avoid entering real names/identifiers. Use coded IDs or de-identified data."
        )

        new_patient_id = st.text_input("Patient ID", key="new_case_patient_id")
        new_name = st.text_input("Patient display name", key="new_case_name")
        new_regimen = st.text_input(
            "Current ART regimen (comma-separated, e.g., TDF, 3TC, EFV)",
            key="new_case_regimen",
        )
        new_encounter_date = st.date_input("Encounter date", key="new_case_encounter_date")

        st.caption("Optional: add one previous clinician note")
        new_prev_note = st.text_area("Previous clinician note", key="new_case_prev_note", height=80)

        st.caption("Optional: add labs")
        st.session_state.setdefault("new_case_labs", [])

        lab_key_map = {
            "Viral load": "viral_load",
            "CD4 count": "cd4",
            "Hemoglobin": "hemoglobin",
            "Serum creatinine": "creatinine",
            "Serum phosphate": "phosphate",
            "Urinalysis": "urinalysis",
            "Random blood sugar": "random_blood_sugar",
        }

        lab_options = list(lab_key_map.keys()) + ["Other"]
        lab_col1, lab_col2 = st.columns(2)
        with lab_col1:
            selected_lab = st.selectbox("Lab test", lab_options, key="new_case_lab_test")
            selected_lab_date = st.date_input("Lab date", key="new_case_lab_date")
        with lab_col2:
            other_lab_name = ""
            if selected_lab == "Other":
                other_lab_name = st.text_input("Lab name", key="new_case_other_lab_name")

            if selected_lab == "Urinalysis":
                selected_lab_value = st.text_area(
                    "Result",
                    key="new_case_lab_value_text",
                    height=80,
                )
            else:
                selected_lab_value = st.text_input("Result", key="new_case_lab_value")

        if st.button("Add lab result"):
            lab_value = (selected_lab_value or "").strip()
            if not lab_value:
                st.error("Lab result is required.")
            elif selected_lab == "Other" and not (other_lab_name or "").strip():
                st.error("Lab name is required for 'Other'.")
            else:
                st.session_state["new_case_labs"] = list(st.session_state.get("new_case_labs") or []) + [
                    {
                        "lab": selected_lab,
                        "date": selected_lab_date.isoformat(),
                        "value": lab_value,
                        "other_name": (other_lab_name or "").strip(),
                    }
                ]

        if st.session_state.get("new_case_labs"):
            st.caption("Added lab results")
            st.json(st.session_state.get("new_case_labs"))
            if st.button("Clear added labs"):
                st.session_state["new_case_labs"] = []

        if st.button("Add case to patient list"):
            pid = (new_patient_id or "").strip()
            name = (new_name or "").strip()
            regimen_items = [r.strip() for r in (new_regimen or "").split(",") if r.strip()]

            if not pid or not name or not regimen_items:
                st.error("Patient ID, display name, and regimen are required.")
            elif any((p.get("patient_id") == pid) for p in (list(patients) + custom_patients)):
                st.error("That Patient ID already exists in the current patient list.")
            else:
                encounter_iso = new_encounter_date.isoformat()

                patient_record: Dict[str, Any] = {
                    "patient_id": pid,
                    "name": name,
                    "art_regimen_current": regimen_items,
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

                # Rationale: allow arbitrary lab entry; preserve existing keys for
                # viral_load/creatinine so current rules continue to work.
                for entry in list(st.session_state.get("new_case_labs") or []):
                    lab_label = str(entry.get("lab") or "").strip()
                    date_iso = str(entry.get("date") or "").strip()
                    raw_value = str(entry.get("value") or "").strip()

                    if not lab_label or not date_iso or not raw_value:
                        continue

                    if lab_label == "Other":
                        lab_key = (entry.get("other_name") or "").strip().lower().replace(" ", "_")
                        if not lab_key:
                            continue
                    else:
                        lab_key = lab_key_map.get(lab_label)
                        if not lab_key:
                            continue

                    patient_record["labs"].setdefault(lab_key, [])

                    if lab_key == "viral_load":
                        try:
                            patient_record["labs"][lab_key].append(
                                {"date": date_iso, "value_copies_per_ml": int(float(raw_value))}
                            )
                        except Exception:
                            patient_record["labs"][lab_key].append({"date": date_iso, "value": raw_value})
                    elif lab_key == "creatinine":
                        try:
                            patient_record["labs"][lab_key].append(
                                {"date": date_iso, "value_umol_per_l": float(raw_value)}
                            )
                        except Exception:
                            patient_record["labs"][lab_key].append({"date": date_iso, "value": raw_value})
                    else:
                        patient_record["labs"][lab_key].append({"date": date_iso, "value": raw_value})

                # Rationale: keep the local list in sync so the newly added patient
                # appears in the list immediately on this rerun.
                custom_patients = custom_patients + [patient_record]
                st.session_state["custom_patients"] = custom_patients
                st.session_state["new_case_labs"] = []
                # Rationale: auto-select the newly added patient to make it clear
                # the add succeeded.
                st.session_state["selected_patient_label"] = f"{pid} - {name}"
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
        top_k = st.slider("Top-K guideline chunks", min_value=1, max_value=10, value=5)
        # Rationale: ensure this is always defined to avoid UnboundLocalError.
        index_max_pages: Optional[int] = 10
        index_pages_choice = st.selectbox(
            "Index max pages",
            options=["All", "10", "20", "50", "100"],
            index=1,
        )
        if index_pages_choice == "All":
            index_max_pages = None
        else:
            index_max_pages = int(index_pages_choice)

        # Rationale: Stage 2 — optional retrieval filter by guideline page range.
        retrieval_page_range: Optional[tuple[int, int]] = None
        retrieval_pages_choice = st.selectbox(
            "Retrieval page range",
            options=["All", "99–114"],
            index=0,
            help="Constrain RAG retrieval to specific guideline pages (uses chunk metadata).",
        )
        if retrieval_pages_choice == "99–114":
            retrieval_page_range = (99, 114)

    with st.expander("LLM settings", expanded=False):
        use_ollama = st.checkbox("Use Ollama for explanations (if available)", value=True)
        llm_mode = st.selectbox(
            "LLM mode",
            ["Checklist only", "Synthesis", "Per-alert explanations"],
            index=0,
        )
        env_model = (os.getenv("OLLAMA_MODEL") or "").strip()
        model_options = ["aadide/medgemma-1.5-4b-it-Q4_K_S", "Custom..."]
        model_choice = st.selectbox("Ollama model", model_options, index=0)
        if model_choice == "Custom...":
            ollama_model_ui = st.text_input("Custom model tag", value=(env_model or ""))
        else:
            ollama_model_ui = model_options[0]
        if env_model:
            st.caption(f"OLLAMA_MODEL is set: {env_model} (UI model selection will be ignored)")
        num_ctx = st.selectbox("Context window (num_ctx)", [2048, 4096, 8192], index=1)

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
        )

    st.subheader("Results")
    if not st.session_state.get("analysis_ran", False):
        st.info("Click 'Save encounter (run checks)' to run the analysis.")
        return

    alerts: List[Alert] = st.session_state.get("analysis_alerts", [])
    status = st.session_state.get("analysis_status")

    if status == "GREEN":
        st.success("GREEN: No alerts detected.")
    else:
        st.warning(f"YELLOW: {len(alerts)} alert(s) to consider.")

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
                if explanation.used_llm:
                    st.caption("Generated with local LLM (Ollama) using retrieved excerpts.")
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


if __name__ == "__main__":
    main()
