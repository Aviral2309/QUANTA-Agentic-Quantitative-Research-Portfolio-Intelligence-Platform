from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
class AuditLogger:
    def __init__(self,path): self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
    def log(self,event:str,payload:dict):
        row={"timestamp":datetime.now(timezone.utc).isoformat(),"event":event,"payload":payload}
        with self.path.open("a",encoding="utf-8") as f: f.write(json.dumps(row,default=str)+"\n")
