"""Streamlit demo entrypoint for the HIV Clinical Nudge Engine.
 
 This app implements an initial local-first “Save vs Finalize” workflow:
 - Save runs deterministic alert rules + RAG retrieval (and optional Ollama explanation)
 - Finalize is blocked until each alert is acknowledged or overridden with a reason
 """
 
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from modules.alert_rules import Alert, run_alerts
from modules.embedder import Embedder, EmbedderConfig
from modules.explanation_generator import ExplanationResult, generate_explanation
from modules.llm_client import OllamaClient
from modules.patient_parser import build_patient_context, load_mock_patients
from modules.rag_engine import RagEngine
from modules.vector_store import VectorSearchResult, create_vector_store


PROJECT_ROOT = Path(__file__).resolve().parent
CONSOLIDATED_GUIDELINE_PDF_PATH = PROJECT_ROOT / "Data" / "Consolidated-HIV-and-AIDS-Guidelines-20230516.pdf"
LEGACY_GUIDELINE_PDF_PATH = PROJECT_ROOT / "Data" / "Uganda Clinical Guidelines 2023.pdf"

# Rationale: prefer the newer consolidated guideline PDF when present, but keep a
# fallback to the legacy filename to avoid breaking existing setups.
GUIDELINE_PDF_PATH = (
    CONSOLIDATED_GUIDELINE_PDF_PATH
    if CONSOLIDATED_GUIDELINE_PDF_PATH.exists()
    else LEGACY_GUIDELINE_PDF_PATH
)
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
 ) -> RagEngine:
     # Rationale: keep a single vector store + embedder + engine per session.
     embedder = _get_embedder(embedding_model_name)
     vector_store = create_vector_store(project_root=PROJECT_ROOT, prefer_chroma=prefer_chroma)
     return RagEngine(
         project_root=PROJECT_ROOT,
         guideline_pdf_path=GUIDELINE_PDF_PATH,
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
     use_ollama: bool,
 ) -> None:
     # Rationale: keep analysis deterministic for alert triggering, and only use the
     # LLM for explanation/citation if available.
 
     patient_copy = copy.deepcopy(patient)
     patient_copy.setdefault("today_encounter", {})
     patient_copy["today_encounter"]["note"] = today_note
 
     context = build_patient_context(patient_copy)
     alerts = run_alerts(context)

     # Rationale: reset per-alert review state for each analysis run. Without this,
     # a previous acknowledgment/override can incorrectly allow finalize on a new
     # Save without review.
     for alert in alerts:
         st.session_state[f"alert_action_{alert.alert_id}"] = "Unreviewed"
         st.session_state[f"alert_override_reason_{alert.alert_id}"] = ""
         st.session_state[f"alert_override_comment_{alert.alert_id}"] = ""
 
     rag = _get_rag_engine(
         prefer_chroma=prefer_chroma,
         embedding_model_name=embedding_model_name,
     )
 
     # Rationale: indexing can be expensive; we allow limiting pages for a faster demo.
     if GUIDELINE_PDF_PATH.exists():
         rag.ensure_indexed(max_pages=index_max_pages)
 
     llm_client = OllamaClient() if use_ollama else None

     ollama_error: Optional[str] = None

     results: List[Dict[str, Any]] = []
     for alert in alerts:
         retrieved = rag.retrieve_for_alert(patient_context=context, alert=alert, top_k=top_k)
         explanation = generate_explanation(
             patient_context=context,
             alert=alert,
             retrieved_chunks=retrieved,
             llm_client=llm_client,
         )

         # Rationale: capture the first LLM failure reason so the UI can show why
         # it fell back to deterministic mode.
         if llm_client is not None and not explanation.used_llm:
             last_err = llm_client.last_error
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
     st.session_state["analysis_ollama_model"] = llm_client.model if llm_client is not None else None
     st.session_state["analysis_ollama_error"] = ollama_error


def main() -> None:
     st.set_page_config(page_title="HIV Clinical Nudge Engine", layout="wide")
     _init_session_state()
 
     st.title("HIV Clinical Nudge Engine (Demo)")
     st.caption(
         "Decision support only. Clinician retains final authority. Synthetic data only."
     )
 
     st.subheader("Guidelines")
     st.write(f"Using local PDF: `{GUIDELINE_PDF_PATH}`")
 
     if GUIDELINE_PDF_PATH.exists():
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
         index_pages_choice = st.selectbox(
             "Index max pages (faster demo)",
             ["All", "20", "50", "100"],
             index=1,
         )
         index_max_pages: Optional[int]
         if index_pages_choice == "All":
             index_max_pages = None
         else:
             index_max_pages = int(index_pages_choice)
 
         use_ollama = st.checkbox("Use Ollama for explanations (if available)", value=True)
 
     save_disabled = not GUIDELINE_PDF_PATH.exists()
     if st.button("Save encounter (run checks)", disabled=save_disabled):
         _run_analysis(
             patient=patient,
             today_note=today_note,
             prefer_chroma=prefer_chroma,
             embedding_model_name=embedding_model_name,
             top_k=top_k,
             index_max_pages=index_max_pages,
             use_ollama=use_ollama,
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
