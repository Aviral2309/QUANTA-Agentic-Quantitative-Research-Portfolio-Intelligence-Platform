from __future__ import annotations
import numpy as np, pandas as pd

def _pct_score(s: pd.Series, higher_is_better: bool=True) -> pd.Series:
    x=pd.to_numeric(s, errors="coerce")
    if x.notna().sum()==0: return pd.Series(50.0,index=s.index)
    scores=x.rank(pct=True, ascending=True)*100
    if not higher_is_better: scores=100-scores+100/max(x.notna().sum(),1)
    return scores.fillna(50).clip(0,100)

def score_stocks(df: pd.DataFrame, weights: dict[str,float]) -> pd.DataFrame:
    x=df.copy()
    directions={"pe_ratio":False,"pb_ratio":False,"ebit":True,"market_cap":True,"volatility":False,"news_sentiment":True,"avg_volume":True}
    total=pd.Series(0.0,index=x.index)
    for factor,w in weights.items():
        if factor not in x.columns: x[factor]=np.nan
        x[f"{factor}_score"]=_pct_score(x[factor],directions.get(factor,True))
        total += w*x[f"{factor}_score"]
    x["factor_score"]=total
    return x.sort_values("factor_score",ascending=False).reset_index(drop=True)
