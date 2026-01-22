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
GUIDELINE_PDF_PATH = PROJECT_ROOT / "Data" / "Uganda Clinical Guidelines 2023.pdf"
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
 
     results: List[Dict[str, Any]] = []
     for alert in alerts:
         retrieved = rag.retrieve_for_alert(patient_context=context, alert=alert, top_k=top_k)
         explanation = generate_explanation(
             patient_context=context,
             alert=alert,
             retrieved_chunks=retrieved,
             llm_client=llm_client,
         )
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
 
     st.subheader("Encounter")
 
     patient_labels = [f"{p.get('patient_id')} - {p.get('name')}" for p in patients]
     selected_label = st.selectbox("Select patient", patient_labels)
     selected_index = patient_labels.index(selected_label)
     patient = patients[selected_index]
 
     today_note_default = (patient.get("today_encounter", {}) or {}).get("note") or ""
     today_note = st.text_area("Today's encounter note", value=today_note_default, height=140)
 
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
                     st.caption("Deterministic fallback explanation (LLM unavailable/disabled).")
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
