import pandas as pd
import random
import os
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import config
import glob

def read_csv_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path).rename(columns=str.lower)

def save_csv_file(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)