# Sample use case: Mr. K.S.

## Source

- Captured from screenshots provided by the project team.

## Patient profile

- **Patient**: Mr. K.S.
- **Age/Sex**: 51-year-old male
- **Occupation**: Long-distance truck driver (Kampala–Mombasa route)
- **ART history**: Initiated on ART 6 years ago
- **Current ART regimen**: TDF/3TC/EFV (fixed-dose combination)
- **Comorbidities**: Essential hypertension (diagnosed 2 years ago), managed on amlodipine 5 mg

## History of presenting complaint

- Presents for a scheduled 6-month refill.
- Feels “fairly well” but reports:
  - Persistent dull aching lower back pain
  - Bilateral “heaviness” in the thighs
- Attributes symptoms to long hours of driving and loading/unloading cargo.
- Pain has become more tiresome over the last 4 months.
- Occasionally uses ibuprofen (over the counter).
- Denies fever, cough, or urinary symptoms.
- Reports strict adherence to ART, but finds it hard to maintain a consistent diet on the road.

## Physical examination

- **General**: Looks well-nourished, not in acute distress.
- **Vitals**:
  - BP: 146/94 mmHg (slightly elevated)
  - HR: 78 bpm
  - Temp: 36.7°C
  - Weight: 74 kg (stable)
- **Musculoskeletal**:
  - Mild tenderness over lumbar paraspinal muscles
  - No focal neurological deficits in lower limbs
  - Gait slightly stiff but steady
- **Other**:
  - No pedal edema
  - Chest is clear

## Clinician’s progress note (as provided)

"Patient seen for routine review. Virologically suppressed and immunologically stable. Complains of chronic musculoskeletal back pain, likely occupational (truck driving). BP is sub-optimally controlled today (146/94); patient admits to occasionally missing Amlodipine while on transit. Trace proteinuria noted on dipstick, likely secondary to hypertension. Creatinine remains within normal limits. Encouraged lifestyle modifications and adherence to antihypertensives. Continue current ART regimen."

## Plan (as provided)

1. Continue TDF/3TC/EFV.
2. Refill Amlodipine 5 mg.
3. PRN ibuprofen for back pain.
4. Review in 6 months.

## Investigation findings (as provided)

- **Viral load**: < 50 copies/mL (Target Not Detected)
- **CD4 count**: 540 cells/mm³
- **Hemoglobin**: 13.4 g/dL
- **Serum creatinine**: 112 µmol/L
- **Serum phosphate**: 0.78 mmol/L
- **Urinalysis**: Protein trace; Glucose negative; Leukocytes negative
- **Random blood sugar**: 6.2 mmol/L

## Draft mapping to current demo patient JSON (for later)

This is a draft mapping to the current app’s expected demo schema (not yet integrated into the UI):

```json
{
  "patient_id": "SAMPLE_MR_KS",
  "name": "Mr. K.S.",
  "art_regimen_current": ["TDF", "3TC", "EFV"],
  "visits": [
    {
      "date": "YYYY-MM-DD",
      "type": "routine",
      "clinician_note": "<paste clinician progress note here>"
    }
  ],
  "labs": {
    "viral_load": [
      { "date": "YYYY-MM-DD", "value_copies_per_ml": 50 }
    ],
    "creatinine": [
      { "date": "YYYY-MM-DD", "value_umol_per_l": 112 }
    ]
  },
  "today_encounter": {
    "date": "YYYY-MM-DD",
    "note": "<paste encounter note here>",
    "orders": [],
    "med_changes": []
  }
}
```
