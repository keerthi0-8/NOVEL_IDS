import pandas as pd, json, hashlib
from pathlib import Path
nsl=Path(r'G:\My Drive\NSL-KDD\NSL_KDD-master\KDDTest+.txt')
unsw=Path(r'G:\My Drive\UNSW-NB15\UNSW_NB15_testing-set.csv')
for name,path,kwargs,col in [('nsl_kdd',nsl,dict(header=None),41),('unsw_nb15',unsw,dict(header=0), 'attack_cat')]:
    print('READ_LABEL_ONLY',name,str(path),flush=True)
    df=pd.read_csv(path,usecols=[col],dtype=str,keep_default_na=False,**kwargs)
    vals=df.iloc[:,0].astype(str).str.strip()
    print(json.dumps({'dataset':name,'rows':len(vals),'labels':sorted(vals.unique().tolist(),key=str.casefold),'label_counts':vals.value_counts().to_dict()},ensure_ascii=False),flush=True)
