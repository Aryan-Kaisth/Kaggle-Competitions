# mlflow_setup.py

import logging
import warnings

import matplotlib.pyplot as plt
import mlflow
import mlflow.catboost
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)
logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="mlflow")
logging.getLogger("torch").setLevel(logging.ERROR)

def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: list, save_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
    )
    plt.title("Out-of-Fold Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_feature_importance(fi_df: pd.DataFrame, save_path: str, top_n: int = 50) -> None:
    top_features = (
        fi_df.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    plot_df = fi_df[fi_df["feature"].isin(top_features)]

    plt.figure(figsize=(10, 8))
    sns.barplot(
        x="importance", y="feature", data=plot_df,
        order=top_features, palette="viridis",
    )
    plt.title(f"Permutation Feature Importance — Top {top_n} (Val Average)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def execute_mlflow_tracking(tracking_data: dict) -> None:
    model_name = tracking_data.get("model_name", "").lower()

    mlflow.set_experiment(tracking_data.get("experiment_name", "Default"))

    with mlflow.start_run(run_name=tracking_data.get("run_name")):

        # params + metadata
        mlflow.log_params(tracking_data.get("params", {}))
        mlflow.log_param("num_folds", tracking_data.get("num_folds"))
        mlflow.log_dict(tracking_data.get("metadata", {}), "metadata/schema.json")

        # global metrics
        mlflow.log_metrics(tracking_data.get("global_metrics", {}))

        # artifacts
        for path in tracking_data.get("artifacts", []):
            artifact_path = "plots" if path.endswith(".png") else "outputs"
            mlflow.log_artifact(path, artifact_path=artifact_path)

        # nested fold runs
        for fold_idx, fold_info in tracking_data.get("folds_data", {}).items():
            with mlflow.start_run(run_name=f"Fold_{fold_idx}", nested=True):

                model_obj = fold_info.get("model")
                if model_name == "xgb":
                    mlflow.xgboost.log_model(model_obj, name="model")
                elif model_name == "lgbm":
                    mlflow.lightgbm.log_model(model_obj, name="model")
                elif model_name == "catboost":
                    mlflow.catboost.log_model(model_obj, name="model")
                else:
                    mlflow.sklearn.log_model(model_obj, name="model")

    print(f"MLflow tracking complete → {tracking_data.get('run_name')}")