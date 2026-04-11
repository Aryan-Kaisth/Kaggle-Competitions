import pandas as pd
import random
import os
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import config
import glob

def read_csv_file(path: str) -> pd.DataFrame:
    print(f"Reading: {path}")
    return pd.read_csv(path).rename(columns=str.lower)

def save_csv_file(df: pd.DataFrame, path: str) -> None:
    print(f"Saving: {path}")
    df.to_csv(path, index=False)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def agnostic_bacc_scorer(clf, X, y):
    # LGBM, XGBoost, ExtraTrees, or Logistic Regression
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)
    else:
        probs = clf.predict(X, prediction_type='Probability')
        
    preds = np.argmax(probs, axis=1)
    return balanced_accuracy_score(y, preds)

def load_predictions(directory: str, file_suffix: str, target_models: list[str] = None) -> dict[str, pd.DataFrame]:
    print(f"Scanning directory: {directory}...")
    
    if target_models:
        files = [os.path.join(directory, f"{name}{file_suffix}") for name in target_models]
    else:
        files = glob.glob(os.path.join(directory, f"*{file_suffix}"))

    predictions = {}
    for file_path in files:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping.")
            continue

        model_name = os.path.basename(file_path).replace(file_suffix, "")
        
        # Load and mathematically align by ID
        df = pd.read_csv(file_path)
        df = df.sort_values("id").reset_index(drop=True)
            
        predictions[model_name] = df
    return predictions