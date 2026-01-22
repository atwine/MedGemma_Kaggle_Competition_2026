# HIV Clinical Nudge Engine (Streamlit Demo)

A local-first clinical decision support demo for HIV care.

This app:
- Runs **deterministic alert rules** on a patient encounter note + structured history.
- Retrieves relevant excerpts from the **Uganda Clinical Guidelines 2023** (local PDF) via a local vector store.
- Generates a clinician-facing explanation using either:
  - A **local Ollama model** (optional), or
  - A deterministic fallback explanation if Ollama is unavailable.

Important: This is **decision support only**. Clinicians retain final authority.

## What’s in this repo

- **Streamlit app**: `app.py`
- **Core modules**: `modules/`
- **Synthetic demo patients**: `Data/mock_patients.json`
- **Local guideline PDF**: `Data/Uganda Clinical Guidelines 2023.pdf`
- **Persistent local vector DB (Chroma)**: `storage/chroma/` (created at runtime)

## Prerequisites

- **Python 3.10+** (the code uses PEP 604 union types like `OllamaConfig | None`) (`modules/llm_client.py:L25`).
- A working Python environment with `pip`.
- (Optional) **Ollama** installed locally for LLM-generated explanations.

## Clone the repository

Choose a local folder where you want the project to live (for example: `C:\Users\<you>\OneDrive\Desktop\`). Then:

```bash
git clone https://github.com/atwine/MedGemma_Kaggle_Competition_2026.git
```

This creates a new folder named `MedGemma_Kaggle_Competition_2026/` containing the app.

## Setup (Windows)

### Option A: Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: PowerShell

```powershell
py -m venv .venv
\.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

### How to use the workflow

1. Select a patient.
2. Edit the **Today’s encounter note** (optional).
3. Click **Save encounter (run checks)**.
4. Review each alert:
   - **Acknowledge**, or
   - **Override** (requires an override reason).
5. **Finalize / Close visit** is disabled until all alerts are reviewed.

## Local LLM (Ollama) setup

The app can use Ollama for explanation generation.

1. Install and start Ollama.
2. Pull the model:

```bash
ollama pull alibayram/medgemma:4b
```

3. Run the app and keep **Use Ollama for explanations (if available)** enabled in the UI.

### Model selection

By default the app uses:
- `alibayram/medgemma:4b` (`modules/llm_client.py:L20-L30`)

To override the model without changing code:

```bash
export OLLAMA_MODEL="<your-model-tag>"
```

On PowerShell:

```powershell
$env:OLLAMA_MODEL = "<your-model-tag>"
```

If Ollama is not available or the model is missing, the app falls back to a deterministic explanation.

## Tests

```bash
pytest
```

## Troubleshooting

### “Guidelines PDF not found”

The app expects the PDF at:
- `Data/Uganda Clinical Guidelines 2023.pdf` (`app.py:L25-L27`)

Place the file there and restart Streamlit.

### First run is slow

The first run may be slow because:
- The embedding model may download on first use.
- The guideline PDF may be chunked and indexed into a local vector store.

You can speed up demos by using the **Index max pages** setting in the app.

### “Ollama unavailable” or no LLM explanation

- Confirm Ollama is installed and running.
- Confirm the model exists:

```bash
ollama list
```

If the model tag differs, set `OLLAMA_MODEL` as shown above.

### Reset the local vector DB

Chroma persists locally under:
- `storage/chroma/` (`modules/vector_store.py:L173-L185`)

To rebuild the index, stop the app and delete `storage/chroma/`.

## Project structure

```text
.
├── app.py
├── Data/
│   ├── mock_patients.json
│   └── Uganda Clinical Guidelines 2023.pdf
├── modules/
│   ├── alert_rules.py
│   ├── embedder.py
│   ├── explanation_generator.py
│   ├── guideline_processor.py
│   ├── llm_client.py
│   ├── patient_parser.py
│   ├── rag_engine.py
│   └── vector_store.py
├── plans/
│   └── implementation-plan.md
├── storage/
│   └── .gitkeep
└── tests/
    ├── test_alert_rules.py
    ├── test_guideline_processor.py
    └── test_rag_engine.py
```
