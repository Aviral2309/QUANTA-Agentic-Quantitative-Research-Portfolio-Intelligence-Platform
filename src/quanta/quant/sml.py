from pathlib import Path
import numpy as np, pandas as pd

def plot_sml(results: pd.DataFrame, rf: float, market_return: float, output: str) -> None:
    import matplotlib.pyplot as plt
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    betas=np.linspace(max(-0.2, results.beta.min()-0.2), results.beta.max()+0.2, 100)
    required=rf+betas*(market_return-rf)
    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(betas,required,label="Security Market Line")
    ax.scatter(results.beta,results.actual_return_annual,s=24,alpha=.7,label="Observed annual return")
    ax.set(xlabel="Beta",ylabel="Annual return",title="QUANTA Security Market Line")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(output,dpi=160); plt.close(fig)
