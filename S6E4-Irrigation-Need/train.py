# train.py

import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, log_loss

import config
from src.folds import create_folds
from src.features import build_features
from src.models import MODELS
from src.utils import read_csv_file, save_csv_file, seed_everything, timer

MODEL_NAME = "logistic"
PARAMS_MAP = {
    "lgbm": config.LGBM_PARAMS,
    "xgb": config.XGB_PARAMS,
    "catboost": config.CATBOOST_PARAMS,
    "histgbm": config.HISTGBM_PARAMS,
    "extratrees": config.EXTRATREES_PARAMS,
    "logistic": config.LOGISTIC_PARAMS
}

def get_feature_cols(df):
    return [c for c in df.columns if c not in config.EXCLUDE_COLS]

def run_training():
    seed_everything(config.SEED)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name  = f"{MODEL_NAME}_{config.RUN}_seed{config.SEED}"

    print(f"\n{'='*50}")
    print(f"Model: {MODEL_NAME}")
    print(f"Folds: {config.N_FOLDS}")
    print(f"Seed: {config.SEED}")
    print(f"Run: {run_name}")
    print(f"{'='*50}\n")

    # Load data
    train_df = read_csv_file(config.RAW_TRAIN)
    test_df  = read_csv_file(config.RAW_TEST)

    train_df.columns = train_df.columns.str.lower()
    test_df.columns  = test_df.columns.str.lower()

    # Target encoding 
    train_df[config.TARGET] = train_df[config.TARGET].map(config.TARGET_MAP)

    # Folds 
    train_df = create_folds(train_df)

    # Features
    num_cols = [
        c for c in train_df.select_dtypes(include="number").columns
        if c not in config.EXCLUDE_COLS
    ]
    cat_cols = [
        c for c in train_df.select_dtypes(include=["object", "category"]).columns
        if c not in config.EXCLUDE_COLS
    ]

    with timer("Feature engineering"):
        train_df = build_features(train_df, num_cols, cat_cols)
        test_df  = build_features(test_df,  num_cols, cat_cols)

    feature_cols = get_feature_cols(train_df)
    n_classes    = len(config.TARGET_MAP)

    # CV Setup
    oof_preds  = np.zeros((len(train_df), n_classes))
    test_preds = np.zeros((len(test_df),  n_classes))
    model_fn = MODELS[MODEL_NAME]
    params = PARAMS_MAP[MODEL_NAME]

    # CV Loop
    for fold in range(config.N_FOLDS):
        print(f"\n--- Fold {fold + 1} / {config.N_FOLDS} ---")

        train_mask = train_df["kfold"] != fold
        valid_mask = train_df["kfold"] == fold

        X_train = train_df.loc[train_mask, feature_cols].copy()
        y_train = train_df.loc[train_mask, config.TARGET].copy()
        X_valid = train_df.loc[valid_mask, feature_cols].copy()
        y_valid = train_df.loc[valid_mask, config.TARGET].copy()
        X_test  = test_df[feature_cols].copy()

        with timer(f"Fold {fold + 1} training"):
            oof_fold, test_fold = model_fn(
                X_train, X_valid, X_test,
                y_train, y_valid,
                params=params,
            )

        oof_preds[valid_mask] = oof_fold
        test_preds+= test_fold / config.N_FOLDS

        # fold score
        fold_score = accuracy_score(
            y_valid,
            np.argmax(oof_fold, axis=1)
        )

        fold_loss = log_loss(
            y_valid,
            oof_fold
        )
        print(f"Fold {fold + 1} Accuracy: {fold_score:.10f}")
        print(f"Fold {fold + 1} Logloss: {fold_loss:.10f}")

    # OOF Score
    oof_score = accuracy_score(
        train_df[config.TARGET],
        np.argmax(oof_preds, axis=1)
    )

    oof_loss = log_loss(
        train_df[config.TARGET],
        oof_preds
    )

    print(f"\n{'='*50}")
    print(f"OOF Accuracy: {oof_score:.10f}")
    print(f"OOF Log Loss: {oof_loss:.10f}")
    print(f"{'='*50}\n")

    # Save OOF probabilities
    class_names = list(config.TARGET_MAP.keys())

    oof_df = pd.DataFrame(oof_preds, columns=class_names)
    oof_df.insert(0, config.ID_COL, train_df[config.ID_COL].values)
    oof_df["true_label"] = train_df[config.TARGET].values
    save_csv_file(oof_df, os.path.join(config.OOF_DIR, f"{run_name}_oof.csv"))

    # Save raw test probabilities
    test_proba_df = pd.DataFrame(test_preds, columns=class_names)
    test_proba_df.insert(0, config.ID_COL, test_df[config.ID_COL].values)
    save_csv_file(
        test_proba_df,
        os.path.join(config.TEST_PROBA_DIR, f"{run_name}_test_proba.csv")
    )

    # Save Kaggle submission
    inv_target_map  = {v: k for k, v in config.TARGET_MAP.items()}
    predicted_class = np.argmax(test_preds, axis=1)
    predicted_label = pd.Series(predicted_class).map(inv_target_map)

    sub_df = pd.DataFrame({
        config.ID_COL: test_df[config.ID_COL].values,
        config.TARGET: predicted_label.values,
    })
    save_csv_file(sub_df, os.path.join(config.SUBMISSIONS_DIR, f"{run_name}_submission.csv"))

if __name__ == "__main__":
    run_training()