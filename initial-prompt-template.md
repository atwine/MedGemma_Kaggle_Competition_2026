# Project Prompt Template
<!-- 
  This is your project brief template. Copy this file and fill it out to define WHAT you want to build.
  The AI will use this to create a comprehensive step-by-step implementation plan.
-->

## 1. High-Level Goal
<!-- 
  **Your Goal:** In one or two sentences, describe the main objective of the project.
  **Example:** "I want to build a web app that converts currency using a public API."
-->

Build a Streamlit-based proof-of-concept HIV Clinical Nudge Engine for the MedGemma Impact Challenge that, after a clinician saves an encounter note, analyzes the patient’s longitudinal history + today’s visit against Uganda Consolidated HIV Guidelines (via RAG over guideline embeddings) and generates actionable, guideline-cited follow-up alerts to reduce missed monitoring and care gaps.

## 2. Core Features & Requirements
<!-- 
  **Your Goal:** List the essential features as a bulleted list. Be specific and detailed.
  **Example:**
  - Must have a dropdown to select the 'from' and 'to' currencies
  - Must have an input box for the amount
  - Must display the converted amount clearly
  - Must show the last updated time for the exchange rate
-->

- Ingest Uganda Consolidated HIV Guidelines (PDF), extract text, chunk into semantically meaningful segments, and generate embeddings for retrieval.
- Store guideline chunks + embeddings in a local vector store for the demo (ChromaDB preferred; fallback to in-memory similarity search if needed).
- Represent a patient with longitudinal data (e.g., 3 years of visits) including:
  - Structured: visits (dates), labs (values + dates), ART regimen history, other meds.
  - Unstructured: clinical notes (free text), including at least one mixed-language example (English + local terms).
- After the clinician clicks "Save" on today’s encounter:
  - Build a unified patient context from history + today’s visit.
  - Retrieve relevant guideline chunks (top_k configurable) using vector similarity.
  - Generate clinician-facing alerts (nudges) that are actionable and traceable to guidelines.
- Alert triage UI:
  - Green: no alerts generated.
  - Yellow: one or more alerts generated.
- Alert traceability requirements:
  - Each alert must show the guideline citation (chunk text excerpt + section/page metadata if available).
  - “Why this alert?” explanation must reference the patient evidence used (e.g., last lab date, med name, symptom mention).
- Encounter workflow attention/acknowledgement:
  - Use a Save vs Finalize approach: Save stores the encounter; Finalize/Close Visit requires the clinician to review/acknowledge all alerts.
  - Allow clinician to acknowledge an alert as "Will address" or "Override".
  - If overridden, require selection of an override reason (dropdown) and optional free-text comment.
- Demo requirements:
  - Provide 3–5 mock patients covering core scenarios (missed monitoring, toxicity signal, DSD eligibility, multilingual note example).
  - Visualize the RAG pipeline steps in the UI (Extract/Parse → Retrieve → Generate).
  - Allow judges to expand and inspect extracted entities and retrieved guideline chunks.
- Safety and messaging:
  - Demo uses synthetic patient data only.
  - UI must include a disclaimer: decision support only; clinician retains final authority.

## 3. Technology Stack
<!-- 
  **Your Goal:** List the programming languages, libraries, and frameworks you want to use.
  **Example:**
  - Language: JavaScript
  - Framework: React
  - Libraries: axios, Material-UI
  - Database: PostgreSQL
-->

- Language: Python 3.9+
- Framework: Streamlit
- Libraries: langchain, sentence-transformers, pandas, numpy
- Database: ChromaDB (local vector database) OR in-memory numpy similarity search fallback

## 4. Code Examples
<!-- 
  **Your Goal:** (Optional but powerful) If you have specific code patterns or styles, 
  create files in the `examples/` folder and reference them here.
  **Example:** "See `examples/my-api-handler.js` for how I want API calls to be structured."
-->

None

## 5. Documentation & References
<!-- 
  **Your Goal:** Provide links to official documentation for any libraries or APIs needed.
  This helps ensure current and correct implementation methods.
  **Example:** 
  - [React Docs](https://react.dev/)
  - [Currency API Docs](https://exchangerate-api.com/docs)
-->

- Uganda Consolidated HIV Guidelines (2023) (local PDF): Data/Uganda Clinical Guidelines 2023.pdf
- MedGemma (official docs): https://developers.google.com/health-ai-developer-foundations/medgemma
- MedGemma local inference (Ollama model): https://ollama.com/MedAIBase/MedGemma1.5
- Streamlit: https://docs.streamlit.io/
- LangChain (RAG): https://docs.langchain.com/oss/python/langchain/rag
- Sentence Transformers: https://www.sbert.net/
- ChromaDB: https://docs.trychroma.com/

## 6. Other Considerations & Gotchas
<!-- 
  **Your Goal:** List anything else important. Tricky parts? Specific constraints? Performance requirements?
  **Example:** "The API has a rate limit of 10 requests per minute, so add error handling for that."
-->

- All patient data must be synthetic; no real PHI.
- Prefer deterministic alerting logic for demo reliability (rules for when an alert triggers), with the LLM/RAG used primarily for explanation and guideline citation.
- Notes can be long; default to summarizing or windowing history (e.g., last 6 months) but allow pulling older context when needed for a specific alert.
- Alert fatigue risk: avoid noisy alerts; keep a small number of high-value alerts for the demo.
- If MedGemma is unavailable, provide a fallback generation strategy (e.g., a smaller local model or prompt-based generation), while preserving guideline citation behavior.

## 7. Success Criteria
<!-- 
  **Your Goal:** Define what "done" looks like. How will you know the project is successful?
  **Example:** 
  - User can convert between any two supported currencies
  - Conversion happens in under 2 seconds
  - Error messages are clear and helpful
-->

- Guidelines PDF is successfully processed into a searchable chunk+embedding store.
- For each demo patient, the system retrieves relevant guideline content and displays it with the alert.
- Green/Yellow status matches expectations for the mock patients.
- Finalize/Close Visit cannot be completed until alerts are reviewed and acknowledged/overridden.
- Demo runs smoothly with acceptable latency (target: a few seconds per post-save analysis on a laptop).

## 8. ML Project Flag (manual)
<!-- 
  Set this flag explicitly to control ML workflow routing. 
  true  -> ML workflows (e.g., TRIPOD+AI pipeline)
  false -> Standard software workflows
  Example values: true | false
-->

is_ml_project: true
