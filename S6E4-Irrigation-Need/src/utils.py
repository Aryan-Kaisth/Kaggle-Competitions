import pandas as pd

def read_csv(path: str) -> pd.DataFrame:
    print(f"Reading: {path}")
    return pd.read_csv(path)

def save_csv(df: pd.DataFrame, path: str) -> None:
    print(f"Saving: {path}")
    df.to_csv(path, index=False)