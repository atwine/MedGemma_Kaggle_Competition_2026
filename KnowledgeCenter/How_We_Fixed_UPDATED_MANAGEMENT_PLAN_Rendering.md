# How we fixed “UPDATED MANAGEMENT PLAN (NEW)” so it shows nicely

## What we wanted

Show **UPDATED MANAGEMENT PLAN (NEW)** as clean, readable sections on the screen.

- If the plan cannot be shown cleanly, **hide the whole section** (so we never show a messy block of text).
- Keep the “raw” plan only behind a collapsed reviewer/debug toggle.

## What was going wrong

### Symptom
The plan section was **hidden** most of the time.

### The real reason
The plan text was often **cut off in the middle**.

So even though we had *some* plan text, it ended early (half-finished). When that happens, the app cannot safely turn it into clean sections, so it hides it.

### Why it was cut off
The local model client (`modules/llm_client.py`) uses a short default output limit:

- `num_predict = 256` (this is the “maximum length” the model is allowed to write)
- Source: `modules/llm_client.py` lines **98–101**

That was too small for a full plan, so the model often stopped early.

## What we changed (the fix)

We fixed it in **two places**:

1) **How we display the plan (UI)**
2) **How we generate the plan (agentic_flow)**

### 1) UI change: show the plan only when it is clean

File: `app.py`

We changed the UI so:

- The plan section is only created when we successfully:
  - read the plan as a proper `{ ... }` object, and
  - build the clean “cards” (Decision / Actions / Monitoring / Counselling)
- If we cannot do that, we show **nothing** for the plan.
- The raw plan is only available under a collapsed section:
  - `View raw plan JSON`

Where to look:

- Standalone plan expander (top-level): `app.py` around **1532–1743**
- Plan inside the `Output` expander: `app.py` around **1830–2006**

### 2) Generation change: handle extra words around the plan

File: `modules/agentic_flow.py`

Sometimes models add extra words like:

- “Here is the plan …”
- then the `{ ... }` content
- then extra words again

We added a helper that:

- finds the **first full `{ ... }` block**, and
- uses only that block

This makes the checker less fragile.

Where to look:

- `_extract_first_json_object(...)`: `agentic_flow.py` around **478–510**
- `_validate_plan_json(...)` now validates the extracted `{ ... }` block: around **512–525**

### 3) Generation change: stop the plan from being cut off

File: `modules/agentic_flow.py`

This was the big win.

Inside `_reason_updated_management_plan(...)` we added:

- a bigger output limit for this plan only:
  - `options_override={"num_predict": 1600}`
- we also asked Ollama for a strict `{ ... }` style reply:
  - `format="json"`

We applied this to:

- the **first plan call**
- the **retry plan call**
- the **organizer/repair call**

Where to look:

- `_PLAN_CHAT_OPTIONS = {"num_predict": 1600}`: around **464–466**
- first plan call uses `format="json"` + `options_override`: around **785–792**
- retry plan call uses the same: around **803–807**
- organizer repair call uses the same: around **654–658**

### 4) Reduce the prompt size (helps avoid cut-offs)

File: `modules/agentic_flow.py`

We also made the “input text” shorter, so the model has more room to finish the plan:

- fewer guideline snippets per subtask
- shorter snippet previews

Where to look:

- evidence snippet reduction: around **742–763**

### 5) Organizer improvement: if the draft is incomplete, rebuild from scratch

File: `modules/agentic_flow.py`

When the plan is cut off, the repair step can fail because the draft is not complete.

We updated the repair instructions to say:

- if the draft is incomplete/invalid, **rebuild a complete plan from scratch**
- keep it concise (max 6 problems)

Where to look:

- `_organize_plan_json(...)` retry feedback: around **622–633**

## How to reproduce / verify (quick checklist)

1) Run the app
2) Enable the agentic plan setting (the sidebar checkbox)
3) Run a patient case
4) Confirm:

- You see **UPDATED MANAGEMENT PLAN (NEW)** with clean sections
- You do **not** see a messy curly-braces text block by default
- Raw plan is only visible inside `View raw plan JSON`

## If it breaks again (what to check first)

- Check if `updated_management_plan_text` is cut off again.
- If it is cut off, increase `_PLAN_CHAT_OPTIONS["num_predict"]` further (for example 2000).
- If the plan still contains extra words, ensure `_extract_first_json_object(...)` is still used before validation.

## Rollback notes

If you need to undo the “long output” change:

- remove the `_PLAN_CHAT_OPTIONS` override
- remove `format="json"` from the three calls

But note: doing that will likely bring back the “cut off mid-way” problem.
