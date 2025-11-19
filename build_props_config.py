#!/usr/bin/env python3
# build_props_config.py
import os, json, sys, glob
from typing import List, Dict
import pandas as pd

MIN_AUC=float(os.getenv("MIN_AUC","0.62"))
MIN_N=int(os.getenv("MIN_N","40"))
MIN_PROBA=float(os.getenv("MIN_PROBA","0.61"))
SHADOW_MIN_PROBA=float(os.getenv("SHADOW_MIN_PROBA","0.58"))
SHADOW_MAX_PROBA=float(os.getenv("SHADOW_MAX_PROBA","0.61"))

USER=os.path.expanduser("~")
ROOTS=[
  os.path.join(USER,"Documents","alpha_signal_engine","data","results","models"),
  os.path.join(USER,"OneDrive","Documents","alpha_signal_engine","data","results","models"),
]

ODDSBIN_CSV="props_winrate_by_odds_bin.csv"     # <= produce from your trainer
PREDICTIONS="predictions_latest_props.csv"      # <= produce from your trainer

DEFAULT_SPORTS=["basketball_nba","icehockey_nhl"]
DEFAULT_MARKETS=["Player Points Over","Player Assists Over","Player Rebounds Over","Shots on Goal Over"]
DEFAULT_BINS=["Fav","Balanced"]

def find_any(root_list: List[str], name: str) -> str|None:
    for r in root_list:
        p=os.path.join(r,name)
        if os.path.isfile(p): return p
    for r in root_list:
        m=glob.glob(os.path.join(r,"**",name), recursive=True)
        if m: return m[0]
    return None

def pick_bins(path:str|None)->List[str]:
    if not path or not os.path.isfile(path): return DEFAULT_BINS
    try:
        df=pd.read_csv(path)
        cols={c.lower():c for c in df.columns}
        df=df[[cols.get("odds_bin"), cols.get("n"), cols.get("win_rate")]].copy()
        df.columns=["odds_bin","n","win_rate"]
        df=df.sort_values("n", ascending=False)
        bins=df[(df.n>=50)&(df.win_rate>=0.52)].odds_bin.tolist()
        bins=[b for b in bins if str(b).lower()!="longshot"]
        return bins or DEFAULT_BINS
    except: return DEFAULT_BINS

def write_cfg(fname:str, sports, markets, bins, min_p, max_p=None):
    cfg={"sports":sports,"markets":markets,"odds_bin":bins,"min_proba":round(min_p,2)}
    if max_p is not None: cfg["max_proba"]=round(max_p,2)
    with open(fname,"w") as f: json.dump(cfg,f,indent=2)
    print(f"✅ wrote {fname}:\n{json.dumps(cfg,indent=2)}")

def main():
    bins=pick_bins(find_any(ROOTS,ODDSBIN_CSV))
    sports=DEFAULT_SPORTS
    markets=DEFAULT_MARKETS

    write_cfg("props_strong_config.json", sports, markets, bins, MIN_PROBA)
    write_cfg("props_shadow_config.json", sports, markets, list({*bins,"Dog"}), SHADOW_MIN_PROBA, SHADOW_MAX_PROBA)

    preds=find_any(ROOTS,PREDICTIONS)
    if preds and os.path.isfile(preds):
        import pandas as pd
        for cfg_name, proba_max in [("props_strong_config.json", None), ("props_shadow_config.json", SHADOW_MAX_PROBA)]:
            cfg=json.load(open(cfg_name))
            df=pd.read_csv(preds)
            cols={c.lower():c for c in df.columns}
            S,M,B,P = cols.get("sport"), cols.get("market"), cols.get("odds_bin"), cols.get("proba")
            if not all([S,M,B,P]): continue
            df0=df.copy()
            steps=[]
            def add(n,x): steps.append((n,int(x)))
            add("start_total",len(df0))
            df1=df0[df0[S].isin(cfg["sports"])]
            add("after_sport",len(df1))
            df2=df1[df1[M].isin(cfg["markets"])]
            add("after_market",len(df2))
            df3=df2[df2[B].isin(cfg["odds_bin"])]
            add("after_odds_bin",len(df3))
            df4=df3[(df3[P]>=cfg["min_proba"]) & ((proba_max is None) or (df3[P]<proba_max))]
            add("after_proba",len(df4))
            import pandas as pd
            pd.DataFrame(steps,columns=["step","count"]).to_csv(f"props_filter_counts{'_shadow' if proba_max else ''}.csv",index=False)
            df4.to_csv(f"predictions_props_candidates{'_shadow' if proba_max else ''}.csv",index=False)
            print(f"Saved survivors → predictions_props_candidates{'_shadow' if proba_max else ''}.csv")
if __name__=="__main__":
    main()
