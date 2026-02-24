"""Debug script: reproduce the repetition bug using the exact sample output.
Writes results to debug_output.txt since terminal capture may not work."""
import json
import sys
import os
import traceback

# Ensure modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output.txt")

def main():
    lines = []
    def log(msg=""):
        lines.append(str(msg))

    log("DEBUG SCRIPT START")

    try:
        from modules.artemis_postprocess import (
            extract_json_array,
            deduplicate_text,
            deduplicate_paragraphs,
            enforce_word_limit,
            clean_artemis_output,
            _split_sentences,
            _similarity,
        )
        log("IMPORTS OK")
    except Exception as e:
        log(f"IMPORT FAILED: {e}")
        log(traceback.format_exc())
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return

    # Shortened but representative sample with repetition
    SAMPLE_MSG = (
        "The patient's creatinine has increased from 0.9 mg/dL (2020) to 1.4 mg/dL (2026). "
        "This represents a 50% increase over 4 years. "
        "The patient is currently on DTG, which was initiated 3 years ago. "
        "The patient's current creatinine is 1.4 mg/dL, which is 140 umol/L. "
        "This value is significantly above the 20 umol/L threshold. "
        "The patient's creatinine rise is 50% over 4 years, which is a significant increase. "
        "The patient's current creatinine is 1.4 mg/dL, which is 140 umol/L. "
        "This value is significantly above the 20 umol/L threshold. "
        "The patient's creatinine rise is 50% over 4 years, which is a significant increase. "
        "The patient's current creatinine is 1.4 mg/dL, which is 140 umol/L. "
        "This value is significantly above the 20 umol/L threshold. "
        "The patient's creatinine rise is 50% over 4 years, which is a significant increase. "
    )

    SAMPLE_RAW = json.dumps([{
        "alert_id": "P005_TEST",
        "title": "Creatinine increase requires investigation",
        "message": SAMPLE_MSG,
        "issue_type": "lab_abnormality",
        "severity": "moderate",
        "recommended_action": "Check creatinine in 1 month.",
        "citations": []
    }])

    # STEP 1: extract_json_array
    log("\n=== STEP 1: extract_json_array ===")
    try:
        alerts = extract_json_array(SAMPLE_RAW)
        log(f"  OK: extracted {len(alerts)} alert(s)")
        log(f"  Message length: {len(alerts[0]['message'])} chars")
    except Exception as e:
        log(f"  FAILED: {type(e).__name__}: {e}")
        log(traceback.format_exc())
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return

    # STEP 2: sentence splitting
    log("\n=== STEP 2: _split_sentences ===")
    msg = alerts[0]["message"]
    sentences = _split_sentences(msg)
    log(f"  Found {len(sentences)} sentences")
    for i, s in enumerate(sentences):
        log(f"  [{i}] ({len(s)} chars): {s[:120]}")

    # STEP 3: similarity between sentences
    log("\n=== STEP 3: sentence similarity ===")
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = _similarity(sentences[i], sentences[j])
            if sim > 0.5:
                log(f"  sim(sent[{i}], sent[{j}]) = {sim:.3f}")

    # STEP 4: deduplicate_text
    log("\n=== STEP 4: deduplicate_text ===")
    deduped = deduplicate_text(msg, similarity_threshold=0.75)
    log(f"  Original: {len(msg)} chars, {len(msg.split())} words")
    log(f"  Deduped:  {len(deduped)} chars, {len(deduped.split())} words")
    log(f"  Deduped text:\n  {deduped}")

    # STEP 5: clean_artemis_output
    log("\n=== STEP 5: clean_artemis_output ===")
    try:
        cleaned = clean_artemis_output(
            SAMPLE_RAW,
            message_max_words=200,
            action_max_words=50,
            sentence_similarity=0.75,
            paragraph_similarity=0.70,
            alert_similarity=0.70,
        )
        log(f"  OK: {len(cleaned)} alert(s)")
        if cleaned:
            c = cleaned[0]
            log(f"  alert_id: {c.get('alert_id')}")
            m = c.get('message', '')
            log(f"  Message: {len(m)} chars, {len(m.split())} words")
            log(f"  Full cleaned message:\n  {m}")
    except Exception as e:
        log(f"  FAILED: {type(e).__name__}: {e}")
        log(traceback.format_exc())

    # STEP 6: _parse_issues_json
    log("\n=== STEP 6: _parse_issues_json ===")
    try:
        from modules.llm_screening import _parse_issues_json
        result = _parse_issues_json(SAMPLE_RAW)
        if result is None:
            log("  RETURNED None (parse_failed)")
        elif len(result) == 0:
            log("  RETURNED [] (parsed_empty)")
        else:
            log(f"  RETURNED {len(result)} alert(s)")
            for r in result:
                m = r.get("message", "")
                log(f"    alert_id={r.get('alert_id')}, msg_len={len(m)} chars, words={len(m.split())}")
                log(f"    message: {m[:300]}")
    except Exception as e:
        log(f"  FAILED: {type(e).__name__}: {e}")
        log(traceback.format_exc())

    log("\nDEBUG SCRIPT END")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main()
