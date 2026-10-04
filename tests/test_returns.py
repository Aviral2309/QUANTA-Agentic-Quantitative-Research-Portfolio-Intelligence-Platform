import pandas as pd
from quanta.quant.returns import calculate_returns, split_train_test

def test_simple_returns():
    p=pd.DataFrame({"A":[100,110,121]}); r=calculate_returns(p)
    assert round(r.A.iloc[1],6)==0.1

def test_split():
    df=pd.DataFrame({"x":range(100)}); a,b=split_train_test(df,.25); assert len(a)==75 and len(b)==25
