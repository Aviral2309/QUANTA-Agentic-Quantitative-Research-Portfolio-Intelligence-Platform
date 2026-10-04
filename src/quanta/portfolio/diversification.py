import pandas as pd

def select_low_correlation_diversifiers(returns: pd.DataFrame, core: list[str], candidates: list[str], n: int=5) -> pd.DataFrame:
    corr=returns[list(dict.fromkeys(core+candidates))].corr()
    rows=[]
    for c in candidates:
        if c in core or c not in corr.index: continue
        vals=corr.loc[c, [k for k in core if k in corr.columns]].dropna()
        if len(vals): rows.append({"ticker":c,"avg_core_correlation":float(vals.mean()),"avg_abs_core_correlation":float(vals.abs().mean())})
    out=pd.DataFrame(rows)
    if out.empty: return out
    return out.sort_values(["avg_abs_core_correlation","avg_core_correlation"]).head(n).reset_index(drop=True)
