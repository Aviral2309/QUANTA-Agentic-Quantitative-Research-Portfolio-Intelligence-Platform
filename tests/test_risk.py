import numpy as np,pandas as pd
from quanta.validation.risk import review_portfolio

def test_concentration_rejected():
    r=pd.DataFrame(np.random.default_rng(0).normal(size=(100,2)),columns=["A","B"])
    review=review_portfolio({"A":.9,"B":.1},r,None,{"max_single_weight":.5,"max_top3_concentration":1,"max_avg_pairwise_correlation":1})
    assert review.status=="REJECT"
