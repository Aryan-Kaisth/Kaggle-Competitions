# train.py
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TORCHDYNAMO_VERBOSE"] = "0"

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, log_loss
import config
from src.mlflow_setup import (
    execute_mlflow_tracking, save_confusion_matrix,
    save_feature_importance
)
from src.features import build_features
from src.folds import create_folds
from src.models import MODELS
from src.utils import read_csv_file, save_csv_file, seed_everything, agnostic_bacc_scorer

MODEL_NAME = "Resnet_RTDL_D"

PARAMS_MAP = {
    # GBDT
    "lgbm": config.LGBM_PARAMS,
    "xgb": config.XGB_PARAMS,
    "catboost": config.CATBOOST_PARAMS,
    "histgbm": config.HISTGBM_PARAMS,
    "extratrees": config.EXTRATREES_PARAMS,

    # Linear
    "logistic": config.LOGISTIC_PARAMS,

    # Pytabkit
    "CatBoost_TD": config.CatBoost_TD_PARAMS,
    "XGB_TD": config.XGB_TD_PARAMS,
    "LGBM_TD": config.LGBM_TD_PARAMS,
    "Resnet_RTDL_D": config.Resnet_RTDL_D_PARAMS,
    "RealMLP_TD": config.RealMLP_TD_PARAMS,
    "TabM_D": config.TabM_D_PARAMS
}

def run_training():
    seed_everything(config.SEED)

    run_name = f"{MODEL_NAME}_{config.RUN}_seed{config.SEED}"
    class_names = list(config.TARGET_MAP.keys())
    n_classes = len(class_names)
    params = PARAMS_MAP[MODEL_NAME]
    model_fn = MODELS[MODEL_NAME]

    print(f"\n{'='*50}")
    print(f"Model: {MODEL_NAME}")
    print(f"Run: {run_name}")
    print(f"Folds: {config.N_FOLDS} | Seed: {config.SEED}")
    print(f"{'='*50}\n")

    # Load data
    train_df = read_csv_file(config.RAW_TRAIN)
    test_df = read_csv_file(config.RAW_TEST)

    # Target encoding and Folds
    train_df[config.TARGET] = train_df[config.TARGET].map(config.TARGET_MAP)
    train_df = create_folds(train_df)

    # Features
    num_cols = [c for c in train_df.select_dtypes(include="number").columns if c not in config.EXCLUDE_COLS]
    cat_cols = [c for c in train_df.select_dtypes(include=["object", "category"]).columns if c not in config.EXCLUDE_COLS]

    train_df = build_features(train_df, num_cols, cat_cols)
    test_df = build_features(test_df,  num_cols, cat_cols)

    feature_cols = [c for c in train_df.columns if c not in config.EXCLUDE_COLS]

    # Initialize MLflow Tracking Dictionary
    tracking_dict = {
        "experiment_name": MODEL_NAME.upper(),
        "run_name": run_name,
        "model_name": MODEL_NAME,
        "num_folds": config.N_FOLDS,
        "params": params,
        "metadata": {
            "n_rows": len(train_df),
            "n_cols": len(feature_cols),
            "feature_names": feature_cols
        },
        "folds_data": {},
        "global_metrics": {},
        "artifacts": []
    }

    # CV Setup
    oof_preds = np.zeros((len(train_df), n_classes))
    test_preds = np.zeros((len(test_df),  n_classes))
    fold_bal_accs = []
    fold_loglosses = []
    fi_records = []

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

        oof_fold, test_fold, model_obj = model_fn(
            X_train, X_valid, X_test,
            y_train, y_valid,
            params=params,
        )

        oof_preds[valid_mask] = oof_fold
        test_preds += test_fold / config.N_FOLDS

        # Fold metrics
        fold_preds = np.argmax(oof_fold, axis=1)
        fold_bacc = balanced_accuracy_score(y_valid, fold_preds)
        fold_ll = log_loss(y_valid, oof_fold)
        fold_bal_accs.append(fold_bacc)
        fold_loglosses.append(fold_ll)

        print(f"Balanced Acc: {fold_bacc:.6f}")
        print(f"Log Loss: {fold_ll:.6f}")

        # Permutation importance on validation set
        perm = permutation_importance(
            model_obj, X_valid, y_valid,
            n_repeats=5,
            random_state=config.SEED,
            scoring=agnostic_bacc_scorer,
            n_jobs=-1,
        )

        for feat, imp in zip(feature_cols, perm.importances_mean):
            fi_records.append({"fold": fold, "feature": feat, "importance": imp})

        # Append to Tracking Dict for this specific fold
        tracking_dict["folds_data"][fold + 1] = {
            "model": model_obj
        }

    # Global OOF Metrics
    oof_preds_class = np.argmax(oof_preds, axis=1)
    oof_bacc = balanced_accuracy_score(train_df[config.TARGET], oof_preds_class)
    oof_ll = log_loss(train_df[config.TARGET], oof_preds)
    std_bacc = float(np.std(fold_bal_accs))
    std_ll = float(np.std(fold_loglosses))

    print(f"\n{'='*50}")
    print(f"OOF Balanced Acc: {oof_bacc:.6f} ± {std_bacc:.6f}")
    print(f"OOF Log Loss: {oof_ll:.6f} ± {std_ll:.6f}")
    print(f"{'='*50}\n")

    # Add global metrics to Tracking Dict
    tracking_dict["global_metrics"] = {
        "oof_balanced_accuracy": oof_bacc,
        "oof_log_loss": oof_ll,
        "std_balanced_accuracy": std_bacc,
        "std_log_loss": std_ll
    }

    # Save Artifacts
    cm_path = os.path.join(config.PLOTS_DIR, f"{run_name}_cm.png")
    save_confusion_matrix(
        train_df[config.TARGET].values,
        oof_preds_class,
        classes=class_names,
        save_path=cm_path,
    )

    fi_df = pd.DataFrame(fi_records)
    fi_path = os.path.join(config.PLOTS_DIR, f"{run_name}_fi.png")
    save_feature_importance(fi_df, save_path=fi_path, top_n=30)

    oof_df = pd.DataFrame(oof_preds, columns=class_names)
    oof_df.insert(0, config.ID_COL, train_df[config.ID_COL].values)
    oof_df["true_label"] = train_df[config.TARGET].values
    oof_path = os.path.join(config.OOF_DIR, f"{run_name}_oof.csv")
    save_csv_file(oof_df, oof_path)

    test_proba_df = pd.DataFrame(test_preds, columns=class_names)
    test_proba_df.insert(0, config.ID_COL, test_df[config.ID_COL].values)
    test_proba_path = os.path.join(config.TEST_PROBA_DIR, f"{run_name}_test_proba.csv")
    save_csv_file(test_proba_df, test_proba_path)

    inv_map = {v: k for k, v in config.TARGET_MAP.items()}
    predicted_label = pd.Series(np.argmax(test_preds, axis=1)).map(inv_map)
    sub_df = pd.DataFrame({
        config.ID_COL: test_df[config.ID_COL].values,
        config.TARGET: predicted_label.values,
    })
    sub_path = os.path.join(config.SUBMISSIONS_DIR, f"{run_name}_submission.csv")
    save_csv_file(sub_df, sub_path)

    # Add all saved paths to Tracking Dict
    tracking_dict["artifacts"] = [cm_path, fi_path, oof_path, test_proba_path, sub_path]

    # Execute MLflow
    execute_mlflow_tracking(tracking_dict)

if __name__ == "__main__":
    run_training()