import importlib.util, json, hashlib
from pathlib import Path
p=Path(r'C:\Users\VaishnaviP\OneDrive\Desktop\Documents\NOVEL_IDS_Project\scripts\phaseC_recovery_execute.py')
spec=importlib.util.spec_from_file_location('phasec_recovery',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
_,m.MAPPINGS=m.load_maps(); report=m.raw_label_audit()
out=Path(r'C:\NOVEL_IDS_runs\phaseC_recovery_20261126\raw_label_vocabulary_audit.json'); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding='utf-8')
print('REPORT',out,'SHA256',hashlib.sha256(out.read_bytes()).hexdigest())
