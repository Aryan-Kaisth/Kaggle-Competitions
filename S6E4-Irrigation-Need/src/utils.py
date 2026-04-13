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