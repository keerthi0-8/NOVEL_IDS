import pandas as pd, json
from pathlib import Path
p=Path(r'G:\My Drive\UNSW-NB15\UNSW_NB15_training-set.csv')
df=pd.read_csv(p,usecols=['attack_cat'],dtype=str,keep_default_na=False)
s=df['attack_cat'].astype(str).str.strip()
print(json.dumps({'path':str(p),'rows':len(s),'labels':sorted(s.unique().tolist(),key=str.casefold),'counts':s.value_counts().to_dict()},ensure_ascii=False),flush=True)
