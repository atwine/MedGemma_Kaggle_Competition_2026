Stage 1 (MedGemma): Receive patient EHR data (which may include more longitudinal data points and not from just one visit). It should pull out past lab results (and date); past key complaints and date; past clinicians' plans (with date); any past regimen (and when switching was done); current regimen; current medications apart from HIV (if any); current labs (with date); doctors plan for this visit. MedGemma should be able to organize these (add any that may be relevant to my goal) and output preferably a JSON.
Stage 2 (RAG): Retrieval based on relevant information from stage 1 output AND check the rule book for common adverse effects, common DDIs, Monitoring schedule for any drug the patient is on (eg is patient is on tenofovir, it should search for Tenofovir in the document). For this we shall focus on the consolidated HIV guidelines, pages 99-114. 
Stage 3 (MedGemma). Synthesize output from stage 1 and stage 2, and loook for any association between patient complaints and the output from stage 2, flag any potential DDIs, identify any symptoms that may suggest drug toxicity. If any are present and not represented in the doctor's plan, nudge the doctor, otherwise allow them to move on. Also nudge if  a prior plan was never executed (e.g., 3 months ago the doctor wrote recheck Cr next visit but it was never done).


Stage 3 prompt if you decide to go with the architecture below. 

# 
"""You are a clinical decision support system for HIV medication safety.

PATIENT INFORMATION (Subjective + Objective):
{stage_1_output}

APPLICABLE GUIDELINES (from retrieval):
{stage_2_output}

TASK: Perform a safety assessment using chain-of-thought reasoning.

Step 1: List all current medications (ARVs + concomitant)
Step 2: For each ARV, check if monitoring is overdue
Step 3: For each ARV + concomitant pair, check for DDIs
Step 4: For each symptom, check if it matches a known toxicity pattern
Step 5: Compare findings against the clinician's documented plan

OUTPUT FORMAT :
- Subjective: Key patient complaints relevant to drug safety
- Objective: Lab values, medication list, monitoring dates
- Assessment: Safety issues identified (with severity)
- Plan: Recommended actions (only if NOT already in clinician's plan)

For each issue in Assessment, specify:
- Issue type: [monitoring_gap | ddi | toxicity_pattern]
- Severity: [high | medium | low]
- Guideline reference: [section number]
- Already addressed in plan: [yes | no]
- Nudge needed: [yes | no]