import pandas as pd
import random
import os
import numpy as np
from sklearn.metrics import balanced_accuracy_score, log_loss, recall_score
import joblib

def read_csv_file(path: str) -> pd.DataFrame:
    print(f"Reading file from: {path}")
    return pd.read_csv(path).rename(columns=str.lower)

def save_csv_file(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Saving file to: {path}")
    df.to_csv(path, index=False)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f'Seed set to {seed}')

def agnostic_bacc_scorer(clf, X, y):
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)
    else:
        probs = clf.predict(X, prediction_type='Probability')

    preds = np.argmax(probs, axis=1)
    
    return balanced_accuracy_score(y, preds)

def compute_bacc(y_true, probs):
    preds = np.argmax(probs, axis=1)
    return balanced_accuracy_score(y_true, preds)

def compute_logloss(y_true, probs):
    return log_loss(y_true, probs)

def compute_recall_per_class(y_true, probs):
    preds = np.argmax(probs, axis=1)
    return recall_score(y_true, preds, average=None)

def load_object(path: str) -> object:
    with open(path, "rb") as file_obj:
        obj = joblib.load(file_obj)
        print(f'Object loaded from {path}')
        return obj

def save_object(obj: object, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file_obj:
        joblib.dump(obj, file_obj)
        print(f'Object saved to {path}')

def reduce_mem_usage(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    if not inplace:
        df = df.copy()
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"Memory before: {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type.name not in numerics:
            continue
            
        has_nan = df[col].isnull().any()
        c_min   = df[col].min()
        c_max   = df[col].max()

        if str(col_type)[:3] == 'int' and not has_nan:
            if   c_min > np.iinfo(np.int8).min  and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)

        else:
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"Memory after: {end_mem:.2f} MB")
    print(f"Reduced by: {start_mem - end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}%)")

    return df