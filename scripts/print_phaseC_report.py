import json
import numpy as np
from pathlib import Path

p = Path(r"C:\NOVEL_IDS_runs\phaseC_recovery_20261126\results.json")
data = json.loads(p.read_text(encoding="utf-8"))

a = data["datasets"]["nsl_kdd"]["test_independent"]["primary"]
b = data["datasets"]["unsw_nb15"]["test_independent"]["primary"]

cm = np.array(a["confusion_matrix"]) + np.array(b["confusion_matrix"])
names = list(a["per_class"])

tp = np.diag(cm)
support = cm.sum(axis=1)
predicted = cm.sum(axis=0)
fp = predicted - tp
fn = support - tp

p = np.divide(tp, tp+fp, out=np.zeros(len(names),float), where=(tp+fp)!=0)
r = np.divide(tp, support, out=np.zeros(len(names),float), where=support!=0)
f = np.divide(2*p*r, p+r, out=np.zeros(len(names),float), where=(p+r)!=0)

active = support > 0
total = support.sum()
accuracy = tp.sum()/total
weights = support/total

print("\n========== NOVEL-IDS | COMBINED EVALUATION ==========")
print(f"Total test samples : {total:,}")
print(f"Classes evaluated : {active.sum()}/12")
print(f"Accuracy           : {accuracy:.2%}")
print(f"Macro Precision    : {p[active].mean():.2%}")
print(f"Macro Recall       : {r[active].mean():.2%}")
print(f"Macro F1           : {f[active].mean():.2%}")
print(f"Weighted Precision : {(p*weights).sum():.2%}")
print(f"Weighted Recall    : {(r*weights).sum():.2%}")
print(f"Weighted F1        : {(f*weights).sum():.2%}")

print("\nCOMBINED PER-CLASS RESULTS")
print(f"{'Class':<20}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>12}")
for i,n in enumerate(names):
    print(f"{n:<20}{p[i]:>11.2%}{r[i]:>12.2%}{f[i]:>12.2%}{support[i]:>12,}")

print("\nMetrics calculated from the sum of both saved confusion matrices.")
print("No retraining or inference performed.")
print("=====================================================\n")
