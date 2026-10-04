import numpy as np,pandas as pd
from quanta.quant.capm import estimate_capm

def test_beta_recovery():
    rng=np.random.default_rng(1); m=pd.Series(rng.normal(.0004,.01,500)); s=.0001+1.5*m+pd.Series(rng.normal(0,.002,500))
    r=estimate_capm("X",s,m,.05); assert abs(r.beta-1.5)<.08
