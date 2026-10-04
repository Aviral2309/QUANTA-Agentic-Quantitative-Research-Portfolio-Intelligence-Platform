from __future__ import annotations
from pathlib import Path
import json, pandas as pd

def ensure_run_dir(base: str, run_id: str) -> Path:
    p=Path(base)/run_id; p.mkdir(parents=True, exist_ok=True); return p

def write_json(path: str|Path, obj) -> None:
    if hasattr(obj,"model_dump"): obj=obj.model_dump()
    Path(path).write_text(json.dumps(obj,indent=2,default=str),encoding="utf-8")

def write_csv(path: str|Path, df: pd.DataFrame) -> None:
    df.to_csv(path,index=True)
