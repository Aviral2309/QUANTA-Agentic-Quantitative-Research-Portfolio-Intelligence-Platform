"""Generate deterministic synthetic data for experimentation; never mix this with real research outputs."""
from pathlib import Path
import numpy as np,pandas as pd
rng=np.random.default_rng(7); n=120
symbols=[f"DEMO{i:03d}.NS" for i in range(n)]
cap=np.exp(rng.normal(23,2,n))
df=pd.DataFrame({"ticker":symbols,"company_name":[f"Demo Company {i}" for i in range(n)],"market_cap":cap})
Path("data/raw").mkdir(parents=True,exist_ok=True); df.to_csv("data/raw/demo_constituents.csv",index=False)
print("Wrote data/raw/demo_constituents.csv (synthetic universe only)")
