# Feb Update Plan (ARTEMIS instruction update)

## Goal (simple)

Add a **new output** to the system that follows the updated ARTEMIS instructions (the “UPDATED MANAGEMENT PLAN” format), **without replacing** any current outputs.

This means:

- The system keeps doing what it already does today.
- We add an additional, new result that can be shown alongside the existing results.

## What we are changing (high level)

The updated instructions add (or strengthen) five things:

1. **Better trend interpretation** (not only “up/down”, but “gradual vs sudden”).
2. **Guideline lookup per drug** using the included guideline documents (multiple sources).
3. AI must produce a **full plan** (not only a list of issues), as a numbered management plan.
4. AI must make a clear decision between **Switch / Hold temporarily / Continue**.
5. AI must do a **self-check** before finalizing (with guideline support for high-impact decisions).

To do that in a “surgical” way, we will only touch the smallest number of places that control those behaviors.

---

## Part 1 — Lab trends: “gradual vs sudden” meaning

### Where this lives today

- File: `modules/patient_parser.py`

This file already computes lab trends and produces summary text.

Right now, the trend logic tells you direction (e.g., “RISING / FALLING / STABLE”), and it prints a timeline.

### What needs to change

Add one additional interpretation layer:

- If the change is **gradual over a long time**, label it “gradual”.
- If the change is a **sudden spike over a short time**, label it “sudden”.

This can be done in a conservative way:

- Only label “gradual” or “sudden” when there is enough data.
- Otherwise label “unknown”.

### Why this must change

The updated ARTEMIS instructions explicitly want the system to tell the difference between:

- A slow change (which may point to a slow developing problem)
- A sudden change (which may point to an acute trigger)

Without this, the system can still show a trend, but it will not follow the updated “Step 2” requirement.

### How you can cross-check

- When the system prints trends, you should see both:
  - Direction (up/down/stable)
  - Pattern type (gradual/sudden/unknown)

---

## Part 2 — Guidelines step (multiple markdown sources)

### Where this lives today

- File(s): guideline documents in `Data\MarkDown Files\` (Markdown format)

### What needs to change

The system must treat Step 3 as:

- “Check all the included reference guidelines.”

This means retrieval should be able to pull drug-specific sections from multiple guideline documents (for example: Uganda, South Africa, WHO), and more Markdown guideline files may be added later.

### Why this must change

Because the updated workflow says the guideline lookup step is central, and it should not be limited to only one guideline source.

### How you can cross-check

- In a test run, retrieval should show evidence coming from more than one guideline document (when those documents exist and are indexed).
- Adding a new Markdown guideline file later should not require code changes (only adding the file).

---

## Part 3 — New output: “UPDATED MANAGEMENT PLAN” (the final structured plan)

### Where the current AI instruction lives today

- File: `modules/agentic_flow.py`
- Function: `_reason()` (this is where the system message for the AI is assembled)

Today, the AI output is shaped like a checklist of issues.

### What needs to change

We add a **new output** that uses a new AI instruction template and produces:

- **UPDATED MANAGEMENT PLAN**
- A list of Problems (Problem 1, Problem 2, ...)
- For each problem:
  - Action (specific)
  - Reason (why)
  - Clinician’s plan: Agree / Disagree / Gap (with explanation)
- Monitoring plan
- Patient counselling

Also:

- The plan should be a **numbered** management plan.
- If the AI is not sure what the specific action should be, it should say **why** it is not sure.

Important: we are not removing the existing output. We add a new output alongside it.

### Why this must change

Because the updated ARTEMIS instructions changed the “main output” from:

- “here are issues”

to:

- “here is the full updated plan”

That is a format and content shift, so it must be implemented in the part that constructs the AI prompt.

### How you can cross-check

- In the code, you should see a new function or new output field that contains the “UPDATED MANAGEMENT PLAN”.
- In a test run, you should see both outputs:
  - existing output (unchanged)
  - new plan output (new)

---

## Part 4 — The “Switch / Hold temporarily / Continue” decision

### Where this will be enforced

- File: `modules/agentic_flow.py`

### What needs to change

The AI must be instructed to always make one of three clear decisions (when applicable):

- Switch
- Hold temporarily
- Continue

This should be required as part of the “UPDATED MANAGEMENT PLAN” output.

### Why this must change

Because the updated instructions say this is a **critical decision point** and the system must not be vague.

### How you can cross-check

- In any output that discusses the main medication decision, the plan should explicitly state one of those three.

---

## Part 5 — Self-check before finalizing

### Where this will be applied

- File: `modules/agentic_flow.py`

### What needs to change

Add the self-check questions from the updated instructions into the AI guidance so the AI re-checks itself before finishing.

We can do this in a safe way:

- The self-check happens “internally” (the AI follows it), but it does not have to print a “self-check section” unless you want it to.

The self-check should be strengthened so that:

- If the AI blames an ARV, it should confirm there is a specific guideline section supporting that.
- If the AI recommends a permanent switch, it should confirm guideline support for a permanent switch vs temporary holding, and consider whether the situation could be acute/reversible.

### Why this must change

Because the updated instructions explicitly require the AI to catch common mistakes before output.

### How you can cross-check

- The self-check questions should be present in the AI instruction text.
- The output should be less likely to:
  - blame the wrong thing
  - make an overly permanent change recommendation
  - ignore parts of the clinician’s plan

---

## Part 6 — Keeping it surgical: we add, we do not replace

### What we will NOT do

- We will not delete the existing output.
- We will not rewrite the whole pipeline.
- We will not change unrelated modules.

### What we WILL do

- Add a new “plan output” result (new output).
- Reuse existing building blocks:
  - existing deterministic checks (`alert_rules.py`)
  - existing trend computation (`patient_parser.py`)
  - existing retrieval and AI call wiring (`agentic_flow.py`)

---

## Minimal files expected to change

- `modules/patient_parser.py`
  - Add “gradual vs sudden” labeling (or compute it next to trend text).

- Retrieval/indexing configuration that points at `Data\MarkDown Files\` so all included Markdown guideline documents are indexed and used.

- `modules/agentic_flow.py`
  - Add a new output that produces the “UPDATED MANAGEMENT PLAN”.
  - Ensure the plan output includes Switch/Hold/Continue.
  - Add self-check instructions.

Possibly (depending on how you want the new output surfaced):

- The place that shows results in the app (so the UI/return structure includes the new output too).

---

## Testing and safety checks (simple)

We should be able to run a small set of demo patient scenarios and see:

- Quick checks trigger without AI.
- Trend output includes gradual/sudden when possible.
- The new plan output appears (and the old output still appears).
- The plan is structured exactly as requested.

---

## Open questions for you to confirm (before any code changes)

1. Should the new output be shown:
   - as a separate section on the UI, or
   - as a separate field in the JSON response, or
   - both?

2. Do you want the self-check to be:
   - internal only (recommended; cleaner output), or
   - printed as a visible section at the end?
