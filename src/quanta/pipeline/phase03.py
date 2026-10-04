from quanta.harness.graph import build_graph

def run_phase03(cfg, phase2: dict, request: str="Build and validate a moderate-risk research portfolio") -> dict:
    graph=build_graph(cfg,phase2["run_dir"])
    state={"run_id":phase2["run_id"],"request":request,"phase2":phase2,"reoptimize_count":0}
    return graph.invoke(state)
