# How the app works

This document explains how to use the **HIV Clinical Nudge Engine (Demo)**.

It is written for a **non-technical** audience (for example: clinicians, program staff, researchers) who want to interact with the app.

## What this app is for

The app helps you **notice important clinical follow-ups** during an HIV care visit.

It does this by:

- Checking the patient’s information (history + today’s note) for specific situations that may need attention.
- Showing you relevant excerpts from the **Uganda Clinical Guidelines 2023** to support the alert.
- Providing a short explanation of *why* the alert was raised.

Important:

- This is **decision support only**.
- It does **not** replace clinical judgement or the official guideline.

## What data is in the demo

The patients in this demo are **synthetic (not real)**.

- You can safely explore the workflow.
- You can edit the “Today’s encounter note” to see how alerts change.

## The main workflow: “Save” vs “Finalize”

The app intentionally separates the visit into two stages:

### 1) Save encounter (run checks)

When you click **Save encounter (run checks)**, the app will:

- Read the selected patient’s information.
- Include anything you typed in **Today’s encounter note**.
- Generate **alerts** if anything needs attention.
- Show supporting information from the guideline.

You can click Save multiple times (for example, after updating the note).

### 2) Finalize / Close visit

The **Finalize / Close visit** button is a safety step.

It will stay disabled until:

- Every alert has been reviewed, and you have either:
  - **Acknowledged** it, or
  - **Overridden** it with a required reason

This helps make sure alerts are not missed.

## Step-by-step: using the app

### Step 1: Confirm the guideline file is found

Near the top, the app shows the local guideline PDF path.

- If it says the guideline PDF is found, you can proceed.
- If it says it is missing, see **Troubleshooting** below.

### Step 2: Select a patient

Use **Select patient** to choose a demo patient.

### Step 3: Review / edit “Today’s encounter note”

The text box **Today’s encounter note** represents what you would type during a visit.

- You can leave it as-is.
- Or add new information (e.g., symptoms, adherence concerns, side effects).

### Step 4: Click “Save encounter (run checks)”

This runs the analysis.

After it runs, you will see:

- A status summary (for example: **GREEN** or **YELLOW**)
- A list of alerts (if any)

### Step 5: Review each alert

Each alert opens as a section.

Inside an alert you will usually see:

- **What the alert is**: a clear message describing the issue.
- **Supporting details**: key patient details the alert is based on.
- **Guideline retrieval**: short references showing which guideline pages were pulled.
- **Why this alert?**: a short explanation.

### Step 6: Acknowledge or Override the alert

For each alert, choose an action:

- **Unreviewed**
  - Default state.
  - Finalize will remain blocked.

- **Acknowledge**
  - Use this if you agree the alert is relevant and you have noted it.

- **Override**
  - Use this if the alert is not applicable to this patient, or you intentionally choose not to follow it now.
  - You must select an **Override reason** (required).
  - You may add an **Override comment** (optional) to document your thinking.

Examples of override reasons include:

- Already addressed
- Not applicable
- Will address later
- Patient declined
- Other

### Step 7: Finalize / Close visit

Once all alerts are acknowledged or overridden (with a reason), the **Finalize / Close visit** button becomes available.

Clicking Finalize marks the visit as completed in the app.

## What “GREEN” and “YELLOW” mean

- **GREEN**
  - No alerts were detected.

- **YELLOW**
  - One or more alerts were detected and should be reviewed before finalizing.

## Advanced settings (optional)

There is an **Advanced settings** section for demo convenience.

You generally do not need to change these, but you can:

- Limit how many guideline pages are processed (useful for a faster demo).
- Turn off the optional local “AI explanation” feature.

## About the optional “AI explanation”

If enabled and available on your computer, the app can generate explanations using a **local Ollama model**.

Key points:

- This is optional.
- If it is not available, the app will still work and will show a simpler fallback explanation.

## Troubleshooting

### The app says the guideline PDF is missing

Make sure the file exists at:

- `Data/Uganda Clinical Guidelines 2023.pdf`

Then restart the app.

### The first run is slow

The first run may take longer because the app may need to prepare the guideline for searching.

For a faster demo:

- In **Advanced settings**, use **Index max pages** to limit how many pages are processed.

### The “AI explanation” is not showing

If the app says it is using a fallback explanation, it usually means the local AI model is not available.

You can:

- Keep using the app normally (it still works).
- Or ask the technical team to confirm Ollama is installed and the model is available.

## Privacy and safety notes

- The demo patients are synthetic.
- Treat all outputs as **supporting information**, not final decisions.
- Always confirm with the official guideline and clinical judgement.
