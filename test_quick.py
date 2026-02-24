import sys, os, json, traceback
sys.path.insert(0, r"c:\Users\ic\OneDrive\Desktop\MegGemma Kaggle Project 2026")
out = r"c:\Users\ic\OneDrive\Desktop\dbg.txt"
try:
    with open(out, "w", encoding="utf-8") as f:
        f.write("START\n")
        from modules.artemis_postprocess import _split_sentences, _similarity, deduplicate_text, clean_artemis_output
        f.write("IMPORTS OK\n")
        
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
        f.write(f"MSG: {len(msg)} chars, {len(msg.split())} words\n")
        
        sents = _split_sentences(msg)
        f.write(f"SENTENCES: {len(sents)}\n")
        for i, s in enumerate(sents):
            f.write(f"  [{i}] {repr(s[:80])}\n")
        
        for i in range(len(sents)):
            for j in range(i+1, len(sents)):
                sim = _similarity(sents[i], sents[j])
                if sim > 0.5:
                    f.write(f"  sim[{i},{j}] = {sim:.3f}\n")
        
        deduped = deduplicate_text(msg, similarity_threshold=0.75)
        f.write(f"DEDUP: {len(deduped)} chars, {len(deduped.split())} words\n")
        f.write(f"DEDUPED: {deduped}\n")
        
        raw = json.dumps([{"alert_id":"T","title":"T","message":msg,"issue_type":"lab","severity":"mod","recommended_action":"Check.","citations":[]}])
        result = clean_artemis_output(raw, message_max_words=200, action_max_words=50)
        f.write(f"PIPELINE: {len(result)} alerts\n")
        if result:
            m = result[0].get("message","")
            f.write(f"CLEAN: {len(m)} chars, {len(m.split())} words\n")
            f.write(f"TEXT: {m}\n")
        f.write("END\n")
except Exception as e:
    with open(out, "w") as f:
        f.write(f"ERROR: {e}\n{traceback.format_exc()}")
