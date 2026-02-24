import sys, os, json, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dbg.txt")
try:
    f = open(out, "w", encoding="utf-8")
    f.write("START\n")
    
    from modules.artemis_postprocess import _split_sentences, _similarity, deduplicate_text, clean_artemis_output
    f.write("IMPORTS OK\n")
    
    # The repeating block from the user's sample
    msg = (
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
    f.write(f"MSG LEN: {len(msg)} chars, {len(msg.split())} words\n")
    
    sents = _split_sentences(msg)
    f.write(f"SENTENCES: {len(sents)}\n")
    for i, s in enumerate(sents):
        f.write(f"  [{i}] {s[:80]}\n")
    
    # Check similarity between sentences
    f.write("\nSIMILARITY:\n")
    for i in range(len(sents)):
        for j in range(i+1, len(sents)):
            sim = _similarity(sents[i], sents[j])
            if sim > 0.5:
                f.write(f"  sim[{i},{j}] = {sim:.3f}\n")
    
    # Test dedup
    deduped = deduplicate_text(msg, similarity_threshold=0.75)
    f.write(f"\nDEDUP: {len(deduped)} chars, {len(deduped.split())} words\n")
    f.write(f"DEDUPED TEXT: {deduped}\n")
    
    # Test full pipeline
    raw = json.dumps([{
        "alert_id": "TEST",
        "title": "Test",
        "message": msg,
        "issue_type": "lab",
        "severity": "moderate",
        "recommended_action": "Check.",
        "citations": []
    }])
    f.write(f"\nRAW JSON LEN: {len(raw)}\n")
    
    result = clean_artemis_output(raw, message_max_words=200, action_max_words=50)
    f.write(f"CLEAN RESULT: {len(result)} alerts\n")
    if result:
        m = result[0].get("message", "")
        f.write(f"CLEAN MSG LEN: {len(m)} chars, {len(m.split())} words\n")
        f.write(f"CLEAN MSG: {m}\n")
    
    f.write("\nDONE\n")
    f.close()
except Exception as e:
    try:
        f.write(f"\nERROR: {e}\n")
        f.write(traceback.format_exc())
        f.close()
    except:
        with open(out, "w") as f2:
            f2.write(f"FATAL: {e}\n{traceback.format_exc()}")
