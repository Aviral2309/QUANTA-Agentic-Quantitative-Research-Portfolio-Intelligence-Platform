from __future__ import annotations
import numpy as np
import pandas as pd
from quanta.validation.advanced_metrics import cagr, sharpe


def block_bootstrap_ci(returns: pd.Series, rf: float=0.0, periods: int=252, samples: int=1000,
                       block_size: int=10, confidence: float=.95, seed: int=42) -> dict:
    s=pd.Series(returns).dropna().to_numpy(float); n=len(s)
    if n<40: return {'status':'INSUFFICIENT_DATA'}
    rng=np.random.default_rng(seed); cs=[]; ss=[]
    for _ in range(samples):
        out=[]
        while len(out)<n:
            start=int(rng.integers(0,max(1,n-block_size+1))); out.extend(s[start:start+block_size])
        x=pd.Series(out[:n]); cs.append(cagr(x,periods)); ss.append(sharpe(x,rf,periods))
    a=(1-confidence)/2
    return {'status':'COMPLETED','samples':samples,'confidence':confidence,
            'cagr_low':float(np.quantile(cs,a)),'cagr_high':float(np.quantile(cs,1-a)),
            'sharpe_low':float(np.quantile(ss,a)),'sharpe_high':float(np.quantile(ss,1-a))}
