import json, hashlib, time
from pathlib import Path
root=Path(r'C:\Users\VaishnaviP\OneDrive\Desktop\Documents\NOVEL_IDS_Project')
run=Path(r'C:\NOVEL_IDS_runs\phaseC_recovery_20261126')
orig=Path(r'C:\NOVEL_IDS_runs\phaseC_one_shot_20261126\freeze_manifest.json')
script=root/'scripts/phaseC_recovery_execute.py'; labelmap=root/'config/phaseC_label_map.yaml'; report=run/'raw_label_vocabulary_audit.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert report.is_file() and not (run/'results.json').exists()
m=json.loads(orig.read_text(encoding='utf-8'))
for p,n in [(script,'phaseC_recovery_evaluator'),(labelmap,'phaseC_evaluation_label_map'),(report,'phaseC_raw_label_vocabulary_audit')]:
 m['files'].append({'path':str(p.resolve()),'sha256':sha(p),'name':n,'bytes':p.stat().st_size})
(run/'freeze_manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
manifest_hash=sha(run/'freeze_manifest.json')
cal=Path(r'C:\NOVEL_IDS_runs\phaseA_primary_seed43_final_retry2\primary_frozen_config.json')
lines=['','## 2026-11-26 — Phase C recovery freeze (pre-inference)','Source: user message, 2026-11-26 evening. Status: frozen; validation-only preflight and raw-label-only vocabulary audit passed.','',
 f'- Recovery evaluator SHA256: `{sha(script)}`.',f'- Evaluation label-map SHA256: `{sha(labelmap)}`.',f'- Primary frozen config SHA256: `{sha(cal)}`.',f'- Raw label-only vocabulary audit SHA256: `{sha(report)}`.',f'- Supplemental recovery freeze-manifest SHA256: `{manifest_hash}`.',
 '- Raw test label audit read only the NSL-KDD label column (22,544 rows, 38 labels) and UNSW-NB15 attack_cat column (82,332 rows, 10 labels); all labels mapped under the pre-approved mapping, including saint -> Probe_Recon. No test feature values were read during this audit.',
 '- Synthetic mapping fixture: 6/6 passed. Validation-only model/baseline preflight: primary plus five ablation checkpoints and selected HGB/MLP baselines passed. No test predictions or metrics produced in preflight.',
 '- The one approved recovery inference run is now authorized by the existing user instruction. Stage 5/6 and original source/test artifacts remain frozen; no model fitting, threshold tuning or taxonomy mutation.']
with (root/'PROGRESS.md').open('a',encoding='utf-8') as f: f.write('\n'.join(lines)+'\n')
print('FROZEN_MANIFEST_SHA256',manifest_hash)
print('RECOVERY_EVALUATOR_SHA256',sha(script))
print('LABEL_MAP_SHA256',sha(labelmap))
print('PRIMARY_CONFIG_SHA256',sha(cal))
print('PROGRESS_APPENDED',root/'PROGRESS.md')
