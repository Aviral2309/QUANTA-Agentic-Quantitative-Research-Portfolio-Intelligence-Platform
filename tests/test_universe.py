import pandas as pd
from quanta.data.universe import select_extremes,normalize_nse_ticker

def test_extremes():
    df=pd.DataFrame({"ticker":[f"S{i}.NS" for i in range(10)],"market_cap":range(1,11)})
    x=select_extremes(df,2,2); assert set(x.ticker)=={"S9.NS","S8.NS","S0.NS","S1.NS"}
def test_ticker(): assert normalize_nse_ticker("reliance")=="RELIANCE.NS"
