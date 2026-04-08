# src/folds.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import config

def create_folds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    df["kfold"] = -1

    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)

    for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df[config.TARGET])):
        df.loc[val_idx, "kfold"] = fold

    print(f"Created {config.N_FOLDS} stratified folds on '{config.TARGET}'")
    return df