from pathlib import Path
p=Path(r"C:\Users\VaishnaviP\OneDrive\Desktop\Documents\NOVEL_IDS_Project\src\novel_ids\inference\api.py")
s=p.read_text(encoding="utf-8")
old='''    output = decisions.reset_index(drop=True)
    output.insert(0, "row_index", np.arange(row_offset, row_offset + n, dtype=np.int64))
'''
new='''    output = decisions.reset_index(drop=True)
    best_indices = probabilities.argmax(axis=1)
    class_names = np.asarray(model.taxonomy.classes, dtype=object)
    output["best_guess_class"] = class_names[best_indices]
    output["best_guess_probability"] = probabilities[np.arange(n), best_indices].astype(np.float32)
    output.insert(0, "row_index", np.arange(row_offset, row_offset + n, dtype=np.int64))
'''
if old not in s: raise SystemExit("prediction insertion anchor not found")
s=s.replace(old,new,1)
old2='''    summary = {
        "rows": int(offset),
'''
new2='''    match_rows = report.feature_matches or []
    level_counts = report.compatibility.get("level_counts", {})
    matched_exact = sum(1 for d in match_rows if int(d.get("level", 3)) == 1 and d.get("relationship") == "equivalent")
    matched_semantic = sum(1 for d in match_rows if int(d.get("level", 3)) == 1 and d.get("relationship") not in {"equivalent", "group_only"})
    group_only = sum(1 for d in match_rows if int(d.get("level", 3)) == 2 or d.get("relationship") == "group_only")
    dataset_specific = sum(1 for d in match_rows if int(d.get("level", 3)) == 3)
    compatibility_score = float((matched_exact + matched_semantic + 0.5 * group_only) / max(1, len(match_rows)))
    trust_level = "High" if compatibility_score >= 0.75 else ("Medium" if compatibility_score >= 0.40 else "Low")
    compatibility_block = {
        "matched_exact": int(matched_exact),
        "matched_semantic": int(matched_semantic),
        "group_only": int(group_only),
        "dataset_specific": int(dataset_specific),
        "compatibility_score": compatibility_score,
        "level_counts": level_counts,
        "score_definition": "(exact + semantic + 0.5*group_only) / max(1, inspected external features)",
    }
    summary = {
        "rows": int(offset),
        "compatibility": compatibility_block,
        "trust_level": trust_level,
'''
if old2 not in s: raise SystemExit("summary insertion anchor not found")
s=s.replace(old2,new2,1)
p.write_text(s,encoding="utf-8")
print("Patched additive best-guess fields and summary compatibility/trust block")
