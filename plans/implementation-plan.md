# Implementation Plan: HIV Clinical Nudge Engine (MegGemma Kaggle Project 2026)

This document is an auto-generated implementation plan. It will be used to guide the development process.

## 1. Core Principles & Success Criteria

- **Primary Goal:** Build a local-first Streamlit proof-of-concept that, after a clinician saves an encounter, analyzes longitudinal HIV patient data + today’s visit against Uganda HIV guidelines (local PDF) using RAG and produces clinician-friendly, guideline-cited alerts.
- **Core Principles:**
  - Deterministic alert triggering (rules) for demo reliability; LLM is used for explanation + guideline citation formatting.
  - Local-first operation: guidelines PDF is local; vector DB local; LLM served locally via Ollama.
  - Traceability: every alert shows the patient evidence used and the retrieved guideline chunk.
  - Safety: synthetic data only; disclaimer in UI.
- **Success Criteria:**
  - Guidelines PDF at `Data/Uganda Clinical Guidelines 2023.pdf` is processed into searchable chunks with metadata.
  - For each demo patient, the app returns Green (no alerts) or Yellow (>=1 alert) as expected.
  - Each alert displays:
    - Patient evidence (e.g., last creatinine date, regimen, symptom mention)
    - Guideline citation (chunk excerpt + location metadata)
  - “Finalize/Close Visit” is blocked until all alerts are acknowledged or overridden with a reason.
  - Post-save analysis completes in a few seconds on a laptop.

## 2. System Architecture & File Structure

```
MegGemma Kaggle Project 2026/
|
├── app.py
├── requirements.txt
├── README.md
├── Data/
│   ├── Uganda Clinical Guidelines 2023.pdf
│   ├── mock_patients.json
│   └── (optional) precomputed_guideline_chunks.json
├── storage/
│   ├── chroma/                  # persisted Chroma data
│   └── cache/                   # optional caching (embeddings, extracted text)
├── modules/
│   ├── guideline_processor.py    # PDF -> text -> chunks (+ metadata)
│   ├── embedder.py               # sentence-transformers wrapper
│   ├── vector_store.py           # Chroma init/query + fallback in-memory
│   ├── patient_parser.py         # normalize structured+unstructured patient context
│   ├── alert_rules.py            # deterministic rule checks -> alerts
│   ├── rag_engine.py             # retrieval orchestration
│   ├── llm_client.py             # Ollama chat client (local)
│   └── explanation_generator.py  # build clinician-facing explanations w/ citations
└── tests/
    ├── test_guideline_processor.py
    ├── test_alert_rules.py
    └── test_rag_engine.py
```

## 3. Implementation Steps

- [ ] **Task 1: Project bootstrap (minimal runnable skeleton)**
  - [ ] Create `app.py`, `modules/`, `tests/`, `storage/`.
  - [ ] Create `requirements.txt` with pinned core deps for a local demo.
  - [ ] Add a `README.md` that documents:
    - how to set up venv
    - how to install deps
    - how to run Streamlit
    - how to run tests

- [ ] **Task 2: Guideline ingestion + chunking**
  - [ ] Implement `modules/guideline_processor.py`:
    - [ ] Extract text from `Data/Uganda Clinical Guidelines 2023.pdf`.
    - [ ] Chunk into semantic-ish blocks (start with simple paragraph/heading heuristics).
    - [ ] Attach metadata per chunk (source filename, page number if available, chunk_id).
  - [ ] Create a small smoke test that returns a non-empty set of chunks.

- [ ] **Task 3: Embeddings + vector store (local)**
  - [ ] Implement `modules/embedder.py` using `sentence-transformers`.
  - [ ] Implement `modules/vector_store.py`:
    - [ ] Local persistent Chroma collection for guideline chunks.
    - [ ] Support rebuild/index command (e.g., a checkbox/button in Streamlit or a CLI function).
    - [ ] Fallback: in-memory cosine similarity if Chroma is unavailable.

- [ ] **Task 4: Patient data model + mock patients**
  - [ ] Define the patient JSON schema in `Data/mock_patients.json`:
    - longitudinal visits
    - meds/regimen history
    - labs with dates
    - free-text notes (include one mixed-language example)
  - [ ] Implement `modules/patient_parser.py` to normalize:
    - “today’s encounter” + “history” into a single context object

- [ ] **Task 5: Deterministic alert rules (Green/Yellow)**
  - [ ] Implement `modules/alert_rules.py` with a minimal set of high-value rules aligned to your demo scenarios:
    - overdue monitoring example (e.g., creatinine overdue while on TDF)
    - toxicity symptom signal example (e.g., EFV + sleep disturbance mentions)
    - DSD eligibility example (stable patient based on VL suppression window)
  - [ ] Each rule must produce an alert object with:
    - id, title, rationale
    - patient evidence payload
    - retrieval query hints (for guideline lookup)

- [ ] **Task 6: RAG retrieval + guideline traceability**
  - [ ] Implement `modules/rag_engine.py`:
    - [ ] Build queries based on patient context + alert rule hints.
    - [ ] Retrieve top_k guideline chunks.
    - [ ] Return chunks with similarity scores + metadata.

- [ ] **Task 7: Local LLM integration (Ollama MedGemma)**
  - [ ] Implement `modules/llm_client.py` calling the local Ollama chat API.
  - [ ] Configure model name (e.g., `MedAIBase/MedGemma1.5:4b`) via env var or Streamlit settings.
  - [ ] Implement `modules/explanation_generator.py`:
    - [ ] Generate clinician-friendly “Why this alert?” text.
    - [ ] Require inclusion of a quoted guideline excerpt in the answer.

- [ ] **Task 8: Streamlit UI (Save vs Finalize workflow)**
  - [ ] Build `app.py` screens:
    - [ ] Patient selector
    - [ ] “Today’s encounter” form (structured fields + free-text note)
    - [ ] Save button: runs analysis and displays Green/Yellow summary + alert list
    - [ ] Alert panel: expandable evidence + retrieved guideline chunks
    - [ ] Acknowledge/Override controls per alert
    - [ ] Finalize button: disabled until all alerts are acknowledged/overridden
    - [ ] Disclaimer banner

- [ ] **Task 9: Tests + minimal validation**
  - [ ] Unit tests for rule triggers (deterministic).
  - [ ] Unit test: guideline chunking returns non-empty chunks.
  - [ ] Unit test: vector retrieval returns top_k chunks.

- [ ] **Task 10: Demo readiness**
  - [ ] Add a “Demo mode” toggle that loads prebuilt mock patients.
  - [ ] Add a “Show pipeline” panel (Parse → Retrieve → Generate) to explain RAG.

## 4. Acceptance Tests (manual)

- [ ] On a mock patient with no issues, Save produces Green and no alerts.
- [ ] On the TDF monitoring patient, Save produces Yellow with an overdue-monitoring alert and guideline citation.
- [ ] Finalize is blocked until each alert is acknowledged or overridden with a reason.
- [ ] “Retrieved guideline chunks” are viewable per alert.

## 5. Notes / Open Questions

- Confirm which exact MedGemma Ollama tag you will standardize on (4B vs other) and the target machine specs.
- Confirm which rules must be in the demo vs later extensions.
