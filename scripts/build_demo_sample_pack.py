from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd,joblib
ROOT=Path(r"C:\Users\VaishnaviP\OneDrive\Desktop\Documents\NOVEL_IDS_Project"); OUT=ROOT/"demo_samples"; OUT.mkdir(exist_ok=True); (OUT/"outputs").mkdir(exist_ok=True)
SEED=42
NSL_NAMES=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty_level"]
def draw(y,n=500,cap=None):
 y=pd.Series(y).astype(str).reset_index(drop=True); counts=y.value_counts(); exact=counts*n/len(y); alloc=np.floor(exact).astype(int); alloc=np.minimum(alloc,counts)
 if cap is not None: alloc=np.minimum(alloc,cap)
 rem=n-int(alloc.sum())
 for cls in (exact-np.floor(exact)).sort_values(ascending=False).index:
  room=int(counts[cls]-alloc[cls]); room=min(room,cap-int(alloc[cls])) if cap is not None else room; k=min(room,rem)
  if k>0: alloc[cls]+=k; rem-=k
  if rem==0: break
 if rem:
  for cls in counts.index:
   room=int(counts[cls]-alloc[cls]); k=min(room,rem)
   if k: alloc[cls]+=k; rem-=k
   if rem==0: break
 rng=np.random.RandomState(SEED); out=[]
 for cls,k in alloc.items():
  if k: out.extend(rng.choice(np.flatnonzero(y.to_numpy()==cls),int(k),replace=False).tolist())
 rng.shuffle(out)
 if len(out)!=n: raise RuntimeError(f"wanted {n}, got {len(out)}")
 return np.asarray(out)
manifest=[]
def save(stem,x,y,source,partition,method,meta):
 y=pd.Series(y).astype(str).reset_index(drop=True); x=x.reset_index(drop=True).copy()
 for suffix,df,has in [("labeled",x.assign(label=y),True),("unlabeled",x,False)]:
  p=OUT/f"{stem}_{suffix}.csv"; df.to_csv(p,index=False)
  manifest.append({"file":p.name,"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"rows":len(df),"columns":len(df.columns),"class_counts":y.value_counts().to_dict(),"has_labels":has,"source":source,"partition":partition,"sampling_method":method,"seed":SEED,**meta})
# NSL active validation only, positions into KDDTrain+
a=joblib.load(ROOT/"data/interim/validation_splits/nsl_kdd_split_v2_no_kddtest_feature_overlap.joblib")
raw=pd.read_csv(r"G:\My Drive\NSL-KDD\NSL_KDD-master\KDDTrain+.txt",header=None,names=NSL_NAMES,dtype=str,keep_default_na=False)
pos=np.asarray(a["validation_source_row_indices"],int); y=pd.Series(a["y_validation"]).astype(str).reset_index(drop=True); rows=raw.iloc[pos].reset_index(drop=True)
if not np.array_equal(rows.label.to_numpy(),y.to_numpy()): raise RuntimeError("NSL validation row/label mismatch")
ix=draw(y); sm=rows.iloc[ix].reset_index(drop=True)
save("nsl_kdd_validation_sample",sm.drop(columns=["label"]),sm.label,"NSL-KDD https://github.com/defcom17/NSL_KDD","Active v2 validation partition, 25,074 rows; derived from KDDTrain+.txt","Seeded stratified by native label, 500 rows, seed 42",{"original_file":"KDDTrain+.txt","validation_artifact":"nsl_kdd_split_v2_no_kddtest_feature_overlap.joblib"})
# UNSW active validation only; project maps this file as canonical source-training partition
b=joblib.load(ROOT/"data/interim/validation_splits/unsw_nb15_split_v2_no_test_feature_overlap.joblib")
up=r"G:\My Drive\UNSW-NB15\UNSW_NB15_testing-set.csv"; ur=pd.read_csv(up,dtype=str,keep_default_na=False,low_memory=False)
upos=np.asarray(b["validation_source_row_indices"],int); urows=ur.iloc[upos].reset_index(drop=True); uy=urows.attack_cat.astype(str)
if not np.array_equal(uy.to_numpy(),pd.Series(b["y_validation"]).astype(str).to_numpy()): raise RuntimeError("UNSW validation row/label mismatch")
ui=draw(uy); us=urows.iloc[ui].reset_index(drop=True)
ux=us.drop(columns=[c for c in ["id","attack_cat","label"] if c in us.columns])
save("unsw_nb15_validation_sample",ux,us.attack_cat,"UNSW-NB15 https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15 ; official page https://research.unsw.edu.au/projects/unsw-nb15-dataset","Active v2 validation partition, 32,999 rows, derived from project-designated UNSW_NB15_testing-set.csv","Seeded stratified by attack_cat, 500 rows, seed 42; id excluded; label=attack_cat",{"original_file":"UNSW_NB15_testing-set.csv","validation_artifact":"unsw_nb15_split_v2_no_test_feature_overlap.joblib"})
# CIC draw from existing 162,276-row sample across 13 available source CSVs
cp=Path(r"C:\NOVEL_IDS_runs\phaseD_cic_partial_20261126\cic_sample.csv"); c=pd.read_csv(cp,low_memory=False)
if "Unnamed: 0" in c.columns: c=c.drop(columns=["Unnamed: 0"])
lc="Label" if "Label" in c.columns else " Label"
if lc not in c: raise RuntimeError("CIC Label missing")
cy=c[lc].astype(str); ci=draw(cy,500,cap=100); cs=c.iloc[ci].reset_index(drop=True); cl=cs[lc].astype(str)
save("cic_ddos2019_sample",cs.drop(columns=[lc]),cl,"Canadian Institute for Cybersecurity CIC-DDoS2019 dataset page; URL to be confirmed by the user","500 rows from existing 162,276-row partial sample spanning 13 available CSV files","Seeded per-class cap (max 100 per raw label), 500 rows, seed 42",{"source_sample":str(cp),"original_files":"13 available CSVs in cic_file_inventory.csv"})
(OUT/"sample_manifest.json").write_text(json.dumps({"seed":SEED,"samples":manifest},indent=2),encoding="utf-8")
readme="""# NOVEL-IDS demo sample pack

These are small excerpts for demonstration only. Raw datasets are not redistributed with the project.

| Sample | Official source | Original file | Partition / method | Rows | Labels | Expected behaviour |
|---|---|---|---|---:|---|---|
| nsl_kdd_validation_sample | NSL-KDD: https://github.com/defcom17/NSL_KDD | KDDTrain+.txt | Active v2 validation (25,074 rows); stratified by native label, seed 42 | 500 | labeled/unlabeled pair | Source-schema compatibility expected; demo only, not test result |
| unsw_nb15_validation_sample | https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15 and https://research.unsw.edu.au/projects/unsw-nb15-dataset | UNSW_NB15_testing-set.csv | Active v2 validation (32,999 rows); stratified by attack_cat, seed 42; id excluded | 500 | labeled/unlabeled pair; label=attack_cat | Source-schema compatibility expected; demo only, not test result |
| cic_ddos2019_sample | Canadian Institute for Cybersecurity CIC-DDoS2019 page; URL to be confirmed by the user | 13 available CSVs represented in existing partial sample | Seeded per-class capped draw from 162,276-row partial sample; cap 100/class | 500 | labeled/unlabeled pair | Unseen dataset; expect low compatibility; see CIC report |

## KDD Cup 99 optional sample
NOT SUPPLIED: data/demo_sources/kddcup.data_10_percent and kddcup.names were not present. No substitute was created. If supplied, attach the 41 feature names in kddcup.names order and strip trailing label dots. UCI Machine Learning Repository, CC BY 4.0: https://archive.ics.uci.edu/dataset/130/kdd+cup+1999+data. This is a same-schema related dataset (NSL-KDD was derived from KDD Cup 99), so results are optimistic and it is NOT an unseen-dataset evaluation.

## Provenance and use
NSL-KDD and UNSW samples use only active validation partitions, never official test or training rows. CIC is an excerpt from the existing partial sample of 13 available files; absent files are not synthesized. Each labeled file adds a column named label; its paired unlabeled file has no label column. Pass --labels-column label to the CLI for labeled evaluation so labels are excluded from onboarding and prediction. Raw values and original feature names are preserved (UNSW id excluded as requested; CIC's generated row-index column is excluded). Manifest includes SHA-256, rows, columns, class counts, provenance, partition, method and seed. Cite NSL-KDD and UNSW-NB15 sources above; cite CIC's official dataset page once confirmed and follow its terms. KDD Cup 99 is CC BY 4.0.
"""
(OUT/"README_DEMO_SAMPLES.md").write_text(readme,encoding="utf-8")
print(json.dumps({"generated":[m["file"] for m in manifest],"nsl_classes":sm.label.value_counts().to_dict(),"unsw_classes":us.attack_cat.value_counts().to_dict(),"cic_classes":cl.value_counts().to_dict(),"kddcup99":"not supplied"},indent=2))



