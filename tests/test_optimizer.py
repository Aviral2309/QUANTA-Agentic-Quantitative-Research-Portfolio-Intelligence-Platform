import numpy as np,pandas as pd
from quanta.portfolio.optimizer import optimize_max_sharpe

def test_optimizer_constraints():
    rng=np.random.default_rng(3); r=pd.DataFrame(rng.normal(.0005,.01,(400,6)),columns=list("ABCDEF"))
    out=optimize_max_sharpe(r,.05,max_weight=.3,restarts=2)
    assert out.success; assert abs(sum(out.weights.values())-1)<1e-6; assert max(out.weights.values())<=.300001
