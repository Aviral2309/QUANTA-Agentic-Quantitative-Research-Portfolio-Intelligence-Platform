import pandas as pd
from quanta.quant.factors import score_stocks

def test_lower_pe_scores_better():
    df=pd.DataFrame({"ticker":["A","B"],"pe_ratio":[10,30]})
    out=score_stocks(df,{"pe_ratio":1.0}).set_index("ticker")
    assert out.loc["A","factor_score"]>out.loc["B","factor_score"]
