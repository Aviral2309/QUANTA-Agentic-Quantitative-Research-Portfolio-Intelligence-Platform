import pandas as pd

def validate_prices(prices: pd.DataFrame, minimum_observations: int) -> list[str]:
    errors=[]
    if prices.empty: return ["price frame is empty"]
    if prices.index.has_duplicates: errors.append("duplicate dates detected")
    if not prices.index.is_monotonic_increasing: errors.append("dates are not sorted")
    for c in prices.columns:
        n=int(prices[c].notna().sum())
        if n < minimum_observations: errors.append(f"{c}: only {n} non-null observations")
    return errors
