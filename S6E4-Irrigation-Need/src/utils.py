import pandas as pd
import random
import os
import numpy as np
import time

def read_csv_file(path: str) -> pd.DataFrame:
    print(f"Reading file from: {path}")
    return pd.read_csv(path).rename(columns=str.lower)

def save_csv_file(df: pd.DataFrame, path: str) -> None:
    print(f"Saving file to: {path}")
    df.to_csv(path, index=False)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f'Seed set to {seed}')