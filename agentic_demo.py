from __future__ import annotations

"""Standalone demo for the agentic RAG flow.

This script runs the agentic planner + per-subtask retrieval in isolation and
prints out what is happening at each step:

- which guideline PDF is indexed (limited pages for speed)
- which patient is used
- what sub-tasks the planner created
- which chunks were retrieved for each sub-task

It does not touch the Streamlit app or the main analysis pipeline.
"""

from pathlib import Path
from typing import List

from modules.embedder import Embedder, EmbedderConfig
from modules.vector_store import create_vector_store, VectorSearchResult
from modules.rag_engine import RagEngine
from modules.patient_parser import load_mock_patients
from modules.agentic_flow import QueryContext, run_agentic_flow
from modules.llm_client import OllamaClient, OllamaConfig


def main() -> None:
    project_root = Path(__file__).resolve().parent

    guideline_pdf = project_root / "Data" / "Consolidated-HIV-and-AIDS-Guidelines-20230516.pdf"
    if not guideline_pdf.exists():
        print(f"[!] Guideline PDF not found at {guideline_pdf}")
        return

    # 1) Initialize embedder + vector store
    print("[1/4] Initializing embedder and vector store (this may take a moment the first time)...")
    embedder = Embedder(EmbedderConfig(model_name="all-MiniLM-L6-v2"))
    # Use in-memory vector store here to keep the demo self-contained.
    vector_store = create_vector_store(
        project_root=project_root,
        prefer_chroma=False,
        collection_name="agentic_demo",
    )

    # 2) Index guideline PDF (limited pages for speed)
    print("[2/4] Indexing guideline PDF (limiting to first 10 pages for faster demo)...")
    engine = RagEngine(
        project_root=project_root,
        guideline_pdf_paths=[guideline_pdf],
        embedder=embedder,
        vector_store=vector_store,
    )
    num_indexed = engine.ensure_indexed(max_pages=100)
    print(f"      Indexed {num_indexed} chunks into the vector store.")

    # 3) Load one mock patient
    print("[3/4] Loading mock patients...")
    patients = load_mock_patients(project_root / "Data" / "mock_patients.json")
    if not patients:
        print("[!] No patients found in Data/mock_patients.json")
        return
    patient = patients[0]
    print(
        f"      Using patient_id={patient.get('patient_id')} "
        f"name={patient.get('name')}"
    )

    # 4) Build QueryContext and run agentic flow
    question = "What do the guidelines say about monitoring and safety for this patient?"
    print("[4/4] Running agentic flow with question:")
    print("      ", question)

    ctx = QueryContext(
        question_text=question,
        patient_raw=patient,
    )

    result = run_agentic_flow(
        ctx,
        embedder=embedder,
        vector_store=vector_store,
        top_k=3,
        page_range=None,
    )

    print("\n=== Planner sub-tasks ===")
    for st in result.debug_info.get("subtasks", []):
        print(f"- name: {st.name!r}")
        print(f"  description: {st.description}")
        print("  retrieval_query:")
        for line in st.retrieval_query.splitlines() or [""]:
            print("    ", line)
        print()

    print("=== Evidence per sub-task ===")
    for bundle in result.debug_info.get("evidence_bundles", []):
        chunks: List[VectorSearchResult] = bundle.chunks
        print(f"- subtask: {bundle.subtask_name!r} ({len(chunks)} chunks)")
        for r in chunks[:2]:
            page = r.metadata.get("page_number")
            print(
                f"    chunk_id={r.chunk_id} page={page} distance={r.distance:.3f}"
            )
            preview = (r.document or "").replace("\n", " ")
            preview = (preview[:200] + "…") if len(preview) > 200 else preview
            print(f"    text≈ {preview!r}")
        if not chunks:
            print("    [no chunks retrieved]")
        print()

    print("=== Final placeholder answer (reasoning not implemented yet) ===")
    print(result.final_answer_text)
    print("\nWarnings:")
    for w in result.warnings:
        print("-", w)

    # Optional: LLM reasoning demo using the retrieved evidence. This uses the
    # existing local Ollama client and remains independent of the Streamlit app.
    print("\n=== LLM reasoning demo (Ollama) ===")
    try:
        llm = OllamaClient(OllamaConfig())
    except Exception as e:  # pragma: no cover - defensive around local setup
        print(f"[!] Could not initialize Ollama client: {type(e).__name__}: {e}")
        llm = None

    if llm is None:
        print("[!] Skipping LLM reasoning because Ollama is unavailable.")
    else:
        evidence_lines: List[str] = []
        for bundle in result.debug_info.get("evidence_bundles", []):
            chunks: List[VectorSearchResult] = bundle.chunks
            if not chunks:
                continue
            evidence_lines.append(f"Subtask {bundle.subtask_name}:")
            for r in chunks[:2]:
                page = r.metadata.get("page_number")
                preview = (r.document or "").replace("\n", " ")
                preview = (preview[:200] + "…") if len(preview) > 200 else preview
                evidence_lines.append(
                    f"- page {page}, chunk {r.chunk_id}: {preview}"
                )
            evidence_lines.append("")

        evidence_text = "\n".join(evidence_lines) or "No evidence retrieved."

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a clinical guideline assistant. Answer the question "
                    "using ONLY the provided guideline excerpts. If the excerpts "
                    "are insufficient or do not cover the case, say that you "
                    "are unsure instead of guessing."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Patient id: {patient.get('patient_id')}, "
                    f"name: {patient.get('name')}\n"
                    f"Question: {question}\n\n"
                    f"Guideline excerpts:\n{evidence_text}"
                ),
            },
        ]

        llm_answer = llm.chat(messages)
        if llm_answer is None:
            print("[!] LLM call failed or returned no content.")
            if llm.last_error:
                print("    Last error:", llm.last_error)
        else:
            print("LLM answer:")
            print(llm_answer)


if __name__ == "__main__":
    main()
