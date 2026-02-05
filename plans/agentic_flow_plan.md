# Agentic RAG Flow Plan (Light Decomposition)

## 1. Objectives and Constraints

- **Goal**: Stabilize and improve answer quality by adding a light agentic layer (planning + task-specific retrieval + verification) on top of the existing RAG stack.
- **Scope**: Design only. No code changes yet – this plan describes *how* we would extend the current system.
- **Constraints**:
  - Reuse existing components where possible: `embedder`, `rag_engine`, `llm_client`, and guideline/alert logic.
  - Avoid invasive architectural rewrites; the agentic layer should sit **on top** of the current query pipeline.
  - Keep the number of new moving parts small and composable.

---

## 2. High-Level Agentic Flow (Conceptual)

End-to-end flow for a single query (user + patient data):

1. **Input Normalization**
   - Receive: free-text question + patient JSON (or similar structured data).
   - Normalize into an internal `QueryContext` object: parsed question, patient summary, and any metadata (e.g., source of patient, flags).

2. **Planning (Planner Agent)**
   - LLM call that reads `QueryContext` and outputs a **structured plan**:
     - Sub-tasks (e.g., "summarize patient", "retrieve guideline rules for condition X", "check contraindications").
     - For each sub-task: a short description and a retrieval focus (what to look for).

3. **Task-Specific Retrieval (per Sub-task)**
   - For each sub-task, construct a focused retrieval query.
   - Use existing `embedder` + `rag_engine` to fetch top-k chunks.
   - Optionally summarize retrieved evidence per sub-task into a compact form.

4. **Global Reasoning (Reasoning Agent)**
   - Single LLM call that combines:
     - `QueryContext` (normalized question + patient summary).
     - Sub-task plans.
     - Retrieved evidence bundles for all sub-tasks.
   - Produces a **draft answer** plus optional intermediate reasoning (not surfaced to end user if not desired).

5. **Verification (Verifier Agent)**
   - Separate LLM call instructed as a **checker**:
     - Takes draft answer, retrieved evidence, and patient summary.
     - Checks for guideline support, contradictions, missing key aspects (e.g., comorbidities, contraindications).
     - Outputs either `OK` or a list of concrete issues + suggested corrections.

6. **Refinement Loop (Optional)**
   - If `Verifier` output is not `OK`, make one refinement pass:
     - Feed draft answer + verifier feedback back into the Reasoning Agent with instructions to correct.
   - Optionally run verifier again for a final sanity check.

7. **Final Answer Assembly**
   - Assemble final textual answer.
   - Optionally attach structured metadata:
     - Which guideline chunks were used.
     - Any residual warnings or uncertainty flags.

---

## 3. Core Data Structures (Conceptual)

These are conceptual; actual implementation details can adapt to existing code.

### 3.1 `QueryContext`

- **Purpose**: Single object passed between steps containing everything about the current interaction.
- **Fields (conceptual)**:
  - `question_text`: string from user.
  - `patient_raw`: original patient JSON or dict.
  - `patient_summary`: normalized summary produced by a summarization step.
  - `metadata`: source, timestamp, flags, etc.

### 3.2 `SubTaskPlan`

- **Purpose**: One unit of work created by the Planner Agent.
- **Fields**:
  - `name`: identifier (e.g., `"retrieve_guideline_rules"`).
  - `description`: 1–3 sentence natural language.
  - `retrieval_query`: focused query string to send to the retriever.
  - `priority`: optional (high/medium/low) for ordering.

### 3.3 `EvidenceBundle`

- **Purpose**: Typed container for all evidence supporting a sub-task.
- **Fields**:
  - `subtask_name`: link back to `SubTaskPlan`.
  - `chunks`: list of retrieved text chunks with IDs, scores, and provenance.
  - `summary`: optional LLM-generated compact summary of the evidence.

### 3.4 `AgenticResult`

- **Purpose**: Final result of the agentic flow, returned to the caller.
- **Fields**:
  - `final_answer_text`.
  - `used_chunks`: list of chunk IDs (for traceability).
  - `warnings`: list of textual warnings or uncertainty reasons.
  - `debug_info` (optional): plan, sub-tasks, verifier feedback, etc., for logs/debug UI.

---

## 4. Agents / Steps and Their Responsibilities

### 4.1 Input Normalization / Patient Summarizer

- **Role**: Turn raw patient JSON + free text into a clinical summary that the other agents can rely on.
- **Inputs**:
  - Raw patient object.
  - Original question.
- **Outputs**:
  - `patient_summary` string or structured fields inside `QueryContext`.
- **Implementation Options**:
  - Pure code (rules-based summarizer) reusing existing alert/guideline logic.
  - Or single LLM call with a strong, deterministic prompt template.

### 4.2 Planner Agent

- **Role**: Decide what needs to be retrieved and reasoned about.
- **Inputs**:
  - `QueryContext` (with `patient_summary`).
- **Outputs**:
  - `List[SubTaskPlan]`.
- **LLM Prompt Characteristics**:
  - Instructed to output **strict JSON** or another machine-parseable format.
  - Example sub-tasks for typical clinical queries:
    - Summarize current problems and risk factors.
    - Retrieve diagnostic criteria and initial screening recommendations.
    - Retrieve management recommendations considering comorbidities.
    - Retrieve contraindications and special precautions.

### 4.3 Task-Specific Retriever

- **Role**: Reuse existing RAG machinery but driven by sub-task-specific queries.
- **Inputs**:
  - `SubTaskPlan.retrieval_query`.
- **Outputs**:
  - `EvidenceBundle`.
- **Technical Notes**:
  - Use existing `embedder` to embed `retrieval_query`.
  - Use existing `rag_engine` to fetch top-k chunks.
  - Enforce a reasonable bound on `k` and token count.
  - Optionally post-process chunks (e.g., drop near-duplicates).

### 4.4 Evidence Summarizer (Optional)

- **Role**: Compress raw chunks into a manageable form for the reasoning agent.
- **Inputs**:
  - `EvidenceBundle.chunks`.
- **Outputs**:
  - Updated `EvidenceBundle.summary`.
- **Technical Notes**:
  - Short LLM call per sub-task; prompt: "Summarize key rules relevant to this patient".
  - Can be skipped initially if token budgets allow passing chunks directly.

### 4.5 Reasoning Agent

- **Role**: Produce a clinical recommendation using all prepared context.
- **Inputs**:
  - `QueryContext`.
  - List of `SubTaskPlan`.
  - List of `EvidenceBundle` (chunks and/or summaries).
- **Outputs**:
  - `draft_answer_text`.
- **LLM Prompt Characteristics**:
  - Clear instructions to:
    - Ground every recommendation in evidence.
    - Explicitly consider comorbidities and contraindications.
    - Prefer stating uncertainty over guessing when evidence is missing.

### 4.6 Verifier Agent

- **Role**: Check the draft answer against evidence and patient data.
- **Inputs**:
  - `draft_answer_text`.
  - `EvidenceBundle` list.
  - `patient_summary` (optional but recommended).
- **Outputs**:
  - `status`: `"OK"` or `"NEEDS_CORRECTION"`.
  - `issues`: list of concrete problems (contradictions, missing rules, unsafe suggestions).
  - Optional suggestion text with how to correct.
- **LLM Prompt Characteristics**:
  - Act as a strict checker, not a re-writer.
  - Be explicit: cite which evidence chunk each issue relates to.

### 4.7 Refinement Step

- **Role**: Incorporate verifier feedback and produce a corrected answer.
- **Inputs**:
  - Original `draft_answer_text`.
  - Verifier feedback (`issues`).
  - Original evidence and `QueryContext`.
- **Outputs**:
  - `final_answer_text`.
- **Policy**:
  - Limit to one correction loop initially to keep latency bounded.

---

## 5. Flow of Events (Step-by-Step)

1. **Receive query and patient**
   - Caller constructs a `QueryContext` with raw inputs.

2. **Normalize patient & question**
   - Optional summarizer step populates `patient_summary`.

3. **Planner Agent call**
   - LLM generates `SubTaskPlan` list.

4. **For each SubTaskPlan**
   - Run retrieval using existing RAG stack.
   - Construct `EvidenceBundle`.
   - Optionally summarize evidence.

5. **Reasoning Agent call**
   - Generate `draft_answer_text` that integrates all sub-tasks.

6. **Verifier Agent call**
   - Evaluate `draft_answer_text` against evidence and patient summary.

7. **Refinement (if needed)**
   - If `NEEDS_CORRECTION`, run one refinement pass via Reasoning Agent.

8. **Assemble AgenticResult**
   - Include final answer, used chunk IDs, warnings, and any debug info.

9. **Return to caller / API layer**
   - Existing API / UI just needs to show `final_answer_text` and optionally some evidence indicators.

---

## 6. Error Handling and Safety Considerations

- **Planner failures**:
  - If planner output is invalid/unparseable → fall back to a default plan with 1–2 generic sub-tasks.
- **Retrieval failures**:
  - If a given sub-task retrieves no chunks → mark sub-task as "no evidence"; Reasoning Agent must treat this as uncertainty, not license to hallucinate.
- **Verifier disagreements**:
  - If verifier repeatedly flags contradictions → system can:
    - Return a conservative answer indicating uncertainty.
    - Surface warnings explicitly to the user.
- **Latency and cost controls**:
  - Cap number of sub-tasks.
  - Cap number of refinement loops.
  - Consider skipping evidence summarization for short docs.

---

## 7. Phased Implementation Roadmap (Non-Binding)

1. **Phase 1 – Skeleton & Interfaces**
   - Define `QueryContext`, `SubTaskPlan`, `EvidenceBundle`, `AgenticResult` (in code terms).
   - Implement a simple orchestrator function that wires steps together but uses stub agents.

2. **Phase 2 – Planner + Task-Specific Retrieval**
   - Implement Planner Agent prompt + parsing.
   - Integrate with existing `embedder` / `rag_engine` for per-sub-task retrieval.

3. **Phase 3 – Reasoning Agent**
   - Implement Reasoning Agent prompt, passing all evidence.
   - Integrate with existing `llm_client` abstraction.

4. **Phase 4 – Verifier Agent and Single Refinement Loop**
   - Implement Verifier prompt and simple pass/fail logic.
   - Add single refinement pass based on verifier feedback.

5. **Phase 5 – Evaluation and Tuning**
   - Compare current (non-agentic) vs agentic pipeline on:
     - Accuracy / guideline adherence.
     - Consistency across repeated queries.
     - Latency and cost.
   - Tune prompts, limits, and planner behaviour based on results.

This plan is intentionally implementation-neutral but technically concrete enough to drive the next step: mapping each conceptual component to actual modules and functions in your repository once you approve the direction.
