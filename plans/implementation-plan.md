# Implementation Plan: HIV Clinical Nudge Engine (MegGemma Kaggle Project 2026)

This document is an auto-generated implementation plan. It will be used to guide the development process.

## 1. Core Principles & Success Criteria

- **Primary Goal:** Build a local-first Streamlit proof-of-concept that, after a clinician saves an encounter, analyzes longitudinal HIV patient data + today’s visit against Uganda HIV guidelines (local PDF) using RAG and produces clinician-friendly, guideline-cited alerts.
- **Core Principles:**
  - Deterministic alert triggering (rules) for demo reliability; LLM is used for explanation + guideline citation formatting.
  - LLM-assisted guideline audit can produce a structured checklist of “things to check in this visit” (labs/monitoring/regimen considerations) grounded in retrieved guideline excerpts; checklist items are treated as alerts that must be acknowledged/overridden, and deterministic rules remain as a fallback/backup for safety and reproducibility.
  - Local-first operation: guidelines PDF is local; vector DB local; LLM served locally via Ollama.
  - Traceability: every alert shows the patient evidence used and the retrieved guideline chunk.
  - Safety: synthetic data only; disclaimer in UI.
- **Success Criteria:**
  - Guidelines PDF at `Data/Consolidated-HIV-and-AIDS-Guidelines-20230516.pdf` is processed into searchable chunks with metadata.
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
│   ├── Consolidated-HIV-and-AIDS-Guidelines-20230516.pdf
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

- [x] **Task 1: Project bootstrap (minimal runnable skeleton)**
  - [x] Create `app.py`, `modules/`, `tests/`, `storage/`.
  - [x] Create `requirements.txt` with pinned core deps for a local demo.
  - [x] Add a `README.md` that documents:
    - how to set up venv
    - how to install deps
    - how to run Streamlit
    - how to run tests

- [x] **Task 2: Guideline ingestion + chunking**
  - [x] Implement `modules/guideline_processor.py`:
    - [x] Extract text from `Data/Consolidated-HIV-and-AIDS-Guidelines-20230516.pdf`.
    - [x] Chunk into semantic-ish blocks (start with simple paragraph/heading heuristics).
    - [x] Attach metadata per chunk (source filename, page number if available, chunk_id).
  - [x] Create a small smoke test that returns a non-empty set of chunks.

- [ ] **Task 3: Embeddings + vector store (local)**
  - [x] Implement `modules/embedder.py` using `sentence-transformers`.
  - [x] Implement `modules/vector_store.py`:
    - [x] Local persistent Chroma collection for guideline chunks.
    - [ ] Support rebuild/index command (e.g., a checkbox/button in Streamlit or a CLI function).
    - [x] Fallback: in-memory cosine similarity if Chroma is unavailable.

- [x] **Task 4: Patient data model + mock patients**
  - [x] Define the patient JSON schema in `Data/mock_patients.json`:
    - longitudinal visits
    - meds/regimen history
    - labs with dates
    - free-text notes (include one mixed-language example)
  - [x] Implement `modules/patient_parser.py` to normalize:
    - “today’s encounter” + “history” into a single context object

- [ ] **Task 5: Deterministic alert rules (Green/Yellow)**
  - [ ] Implement `modules/alert_rules.py` with a minimal set of high-value rules aligned to your demo scenarios:
    - overdue monitoring example (e.g., creatinine overdue while on TDF)
    - toxicity symptom signal example (e.g., EFV + sleep disturbance mentions)
    - DSD eligibility example (stable patient based on VL suppression window)
  - [x] Each rule must produce an alert object with:
    - id, title, rationale
    - patient evidence payload
    - retrieval query hints (for guideline lookup)

- [x] **Task 6: RAG retrieval + guideline traceability**
  - [x] Implement `modules/rag_engine.py`:
    - [x] Build queries based on patient context + alert rule hints.
    - [x] Retrieve top_k guideline chunks.
    - [x] Return chunks with similarity scores + metadata.

- [x] **Task 7: Local LLM integration (Ollama MedGemma)**
  - [x] Implement `modules/llm_client.py` calling the local Ollama chat API.
  - [x] Configure model name (e.g., `MedAIBase/MedGemma1.5:4b`) via env var or Streamlit settings.
  - [x] Implement `modules/explanation_generator.py`:
    - [x] Generate clinician-friendly “Why this alert?” text.
    - [x] Require inclusion of a quoted guideline excerpt in the answer.

- [x] **Task 8: Streamlit UI (Save vs Finalize workflow)**
  - [x] Build `app.py` screens:
    - [x] Patient selector
    - [x] “Today’s encounter” form (structured fields + free-text note)
    - [x] Save button: runs analysis and displays Green/Yellow summary + alert list
    - [x] Alert panel: expandable evidence + retrieved guideline chunks
    - [x] Acknowledge/Override controls per alert
    - [x] Finalize button: disabled until all alerts are acknowledged/overridden
    - [x] Disclaimer banner

- [x] **Task 9: Tests + minimal validation**
  - [x] Unit tests for rule triggers (deterministic).
  - [x] Unit test: guideline chunking returns non-empty chunks.
  - [x] Unit test: vector retrieval returns top_k chunks.

- [ ] **Task 10: Demo readiness**
  - [ ] Add a “Demo mode” toggle that loads prebuilt mock patients.
  - [ ] Add a “Show pipeline” panel (Parse → Retrieve → Generate) to explain RAG.
  - [ ] Include an “Audit checklist output” example in the pipeline panel (what the model checked + which guideline excerpts were used).

- [ ] **Task 11: Pilot / live-readiness (real data testing)**
  - **Goal:** Allow colleagues to test the app using realistic patient scenarios (real or de-identified) while protecting privacy and maintaining safety.
  - **Why this is needed:**
    - Demo JSON is useful for development, but a pilot needs a supported way to bring in real patient data and to record what was reviewed/overridden.
    - Clinical use requires access control and an audit trail.

  - [ ] **11.1 Real patient data ingestion (replace demo JSON only at the boundary)**
    - [ ] Add a new patient data source option in the UI:
      - [ ] **Option A (recommended first):** upload/import a file (JSON/CSV) for testing.
      - [ ] **Option B (later):** connect to a live system (e.g., EMR) via an API.
    - [x] Add a simple “Add patient case” form in the UI for manual entry (supports quick colleague testing without editing JSON).
    - [ ] Define and document a “live patient” input schema (minimal fields required by rules):
      - [ ] patient identifier (or test ID), encounter date, current regimen
      - [ ] labs: viral load (date/value), creatinine (date/value) where relevant
      - [ ] notes text (previous notes + today note)
    - [ ] Add validation with clear user-facing errors when required fields are missing.
    - [ ] Add a “safe demo mode” option that forces synthetic/de-identified data only.

  - [ ] **11.2 Privacy + access control**
    - [ ] Add basic authentication (password-protected access) for pilot sites.
    - [ ] Add a prominent reminder banner when real patient data is being used.
    - [ ] Document operational guidance:
      - [ ] run on a clinic laptop or internal network
      - [ ] do not expose Streamlit publicly

  - [ ] **11.3 Audit trail (who reviewed what, and why)**
    - [ ] Persist an audit record for each analysis run, including:
      - [ ] timestamp, patient ID (or test ID), which alerts triggered
      - [ ] per-alert action (Acknowledged/Overridden), override reason, optional comment
    - [ ] Rationale: pilot evaluation needs traceability and post-hoc review.
    - [ ] Storage approach (minimal): local SQLite file under `storage/`.

  - [ ] **11.4 Welcome / navigation page (user friendliness)**
    - [ ] Add a first screen that explains:
      - [ ] what the tool does and does not do
      - [ ] how to use Save vs Finalize
      - [ ] how to interpret GREEN vs YELLOW
      - [ ] privacy reminder when using real data
    - [ ] Rationale: reduces onboarding time for new clinical users.

  - [ ] **11.5 Pilot validation and safety checks**
    - [ ] Create a set of 5–10 “pilot scenarios” (clinical vignettes) and expected outcomes.
    - [ ] Add a checklist for pilot testers:
      - [ ] false positives/false negatives noted
      - [ ] whether guideline citations were relevant
      - [ ] whether the workflow is usable (time, clarity)

  - [ ] **11.6 MedGemma innovation layer (optional, but recommended for the challenge)**
    - **Goal:** Use MedGemma in a clearly valuable way while keeping alert triggering reproducible and testable.
    - **Why this is needed:** The deterministic rules provide stability for testing, while MedGemma can add medical understanding, better guidance, and faster expansion of rule coverage.

    - [ ] **11.6.1 Note → structured clinical signals (supports real-world notes)**
      - [ ] Use MedGemma to extract key signals from free-text notes (e.g., symptoms, side effects, adherence concerns).
      - [ ] Display extracted signals to the user for transparency.
      - [ ] Feed extracted signals into deterministic rules (rules decide the trigger; model helps interpret text).
      - [ ] Rationale: real patient scenarios often include key details only in notes; this improves usefulness without losing reproducibility.

    - [ ] **11.6.2 Guideline-grounded guidance (questions, urgency, plausible reasons)**
      - [ ] For each triggered alert, use MedGemma to generate:
        - [ ] plausible reasons / differential considerations
        - [ ] what to ask next (targeted questions)
        - [ ] urgency / suggested timeframe (when appropriate)
      - [ ] Require that each suggestion is grounded in retrieved guideline excerpts (quoted + cited).
      - [ ] Rationale: increases clinical value beyond “an alert exists”, while staying evidence-based.

    - [ ] **11.6.4 Guideline audit checklist (LLM-first, rules as backup)**
      - [ ] Use MedGemma + retrieved guideline excerpts to produce a structured checklist for the current visit, e.g.:
        - [ ] recommended checks/labs to consider based on today’s note + longitudinal history
        - [ ] monitoring interval reminders (where supported by retrieved guideline excerpts)
        - [ ] questions to clarify missing information before acting
      - [ ] Require that each checklist item includes a quoted + cited guideline excerpt (or explicitly states that guideline evidence was not retrieved).
      - [ ] Default to a safe fallback response when retrieval is empty/low-quality (do not “make up” guideline recommendations).
      - [ ] Present checklist items as alert-like items in the UI, requiring Acknowledge/Override (with a reason) before Finalize.
      - [ ] Keep deterministic alert rules as a backup layer for high-value safety checks and reproducibility.
      - [ ] Rationale: matches clinician expectations (a visit checklist) while retaining a deterministic safety net.

    - [ ] **11.6.3 Model-proposed new rules with clinician approval (rule growth over time)**
      - [ ] Add a workflow where, when a new scenario arises, MedGemma can propose a candidate rule:
        - [ ] proposed rule name + trigger criteria
        - [ ] patient data fields required
        - [ ] linked guideline excerpt(s) supporting it
        - [ ] suggested test cases (positive/negative examples)
      - [ ] Require human approval before the rule becomes active for future patients.
      - [ ] Store approved rules in a versioned format (e.g., JSON/YAML) so changes are reviewable.
      - [ ] Rationale: enables rapid expansion of coverage while preserving safety and auditability.

  - [ ] **11.7 Deployment approach (pilot)**
    - [ ] Document 2 supported ways to run:
      - [ ] Local laptop mode (simplest)
      - [ ] Small on-prem server mode (shared within a clinic network)
    - [ ] Define environment requirements (Python, model availability, guideline PDF path).

- [ ] **Task 12: MedGemma Stage 1–3 Safety Pipeline (MVP)**
  - **Goal:** Integrate a laptop-friendly version of the colleague’s Stage 1–3 design without disrupting existing deterministic rules or UI workflow.
  - [ ] **12.1 Stage 1 (deterministic JSON extraction)**
    - [ ] Produce a structured JSON summary from the patient record (no LLM), including:
      - [ ] past labs with dates (extend beyond creatinine/VL as data allows)
      - [ ] past key complaints with dates (light heuristics from notes)
      - [ ] past clinician plans with dates (light heuristics from notes)
      - [ ] regimen history with switch dates (if available)
      - [ ] current regimen and current non-HIV medications (if available)
      - [ ] current labs with dates
      - [ ] today’s plan (from today note/orders)
    - [ ] Mark missing fields explicitly; keep schema stable for downstream stages.
  - [ ] **12.2 Stage 2 (RAG evidence bundle)**
    - [ ] Drive retrieval from Stage 1 signals (e.g., per active drug or drug class).
    - [ ] Add an optional page-range filter for consolidated HIV guideline pages 99–114 using chunk metadata.
    - [ ] Keep `top_k` small (2–4) and return a compact evidence bundle (chunk_id, page_number, quote).
  - [ ] **12.3 Stage 3 (single-call LLM synthesis → structured issues)**
    - [ ] Create one LLM prompt that ingests Stage 1 JSON + Stage 2 evidence bundle and outputs JSON issues with fields:
      - [ ] `issue_type` [monitoring_gap | ddi | toxicity_pattern]
      - [ ] `severity` [high | medium | low]
      - [ ] `guideline_reference` (section/page)
      - [ ] `already_in_plan` [yes | no]
      - [ ] `nudge_needed` [yes | no]
    - [ ] Map issues to existing `Alert` objects for the same ack/override gating.
  - [ ] **12.4 LLM call management (latency control)**
    - [ ] Add an "LLM mode" selector in UI with:
      - [ ] Checklist only (1 call; current audit checklist)
      - [ ] Synthesis (1 call; Stage 3 issues)
      - [ ] Per‑alert explanations (N calls; mark as slow)
    - [ ] Default to Checklist only; keep deterministic rules always active.
  - [ ] **12.5 LLM settings UI**
    - [ ] Add model selector (default `aadide/medgemma-1.5-4b-it-Q4_K_S`), with a "Custom" input; respect `OLLAMA_MODEL` env override and show a notice when active.
    - [ ] Add `num_ctx` selector (e.g., 2048/4096/8192) and pass to Ollama `options` alongside `num_predict`.
  - [ ] **12.6 Performance & GPU guidance (dev notes)**
    - [ ] Keep RAG `top_k` small and excerpt truncation in prompts.
    - [ ] Cap `num_predict` (e.g., 256) and prefer a single LLM call per Save.
    - [ ] Prefer small quantized models; verify GPU usage via `ollama ps`.
    - [ ] Consider streaming later for UX; keep timeout safeguards.

## 4. Acceptance Tests (manual)

- [ ] On a mock patient with no issues, Save produces Green and no alerts.
- [ ] On the TDF monitoring patient, Save produces Yellow with an overdue-monitoring alert and guideline citation.
- [ ] On a case with a symptom mention in today’s note, the app produces a guideline-cited “things to check” checklist item (or a safe fallback if no relevant guideline excerpts are retrieved).
- [ ] Finalize is blocked until each alert is acknowledged or overridden with a reason.
- [ ] “Retrieved guideline chunks” are viewable per alert.
 - [ ] With "Page range 99–114" enabled, retrieval returns only chunks within that range (or a safe empty result).
 - [ ] Stage 3 synthesis (LLM mode=Synthesis) returns structured JSON issues mapped to Alerts.
 - [ ] A DDI scenario surfaces a `ddi` issue with a guideline reference and a nudge only if not already in clinician’s plan.
 - [ ] A missed prior plan (e.g., “recheck Cr next visit”) is detected and nudged if not executed.
 - [ ] Switching LLM modes changes the number of LLM calls per Save accordingly (Checklist: 1, Synthesis: 1, Per‑alert: N).

## 5. Notes / Open Questions

- Confirm which exact MedGemma Ollama tag you will standardize on (4B vs other) and the target machine specs.
- Confirm which rules must be in the demo vs later extensions.

- Confirm the pilot testing approach:
  - File-based import of real/de-identified cases vs direct EMR integration.
  - Whether the pilot is read-only (“shadow mode”) or must write back notes/actions.
