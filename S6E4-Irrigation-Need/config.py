# config.py
import os

# --- Paths ---
RAW_TRAIN = os.path.join("data", "raw", "train.csv")
RAW_TEST = os.path.join("data", "raw", "test.csv")
TRAIN_PATH = os.path.join("data", "processed", "train_folded.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")

OOF_DIR = os.path.join("artifacts", "oof")
SUBMISSIONS_DIR = os.path.join("artifacts", "submissions")
TEST_PROBA_DIR = os.path.join("artifacts", "test_proba")
PLOTS_DIR = os.path.join("artifacts", "plots_dump")

# --- Extras ---
TARGET       = "irrigation_need"
TARGET_MAP   = {"Low": 0, "Medium": 1, "High": 2}
ID_COL       = "id"
EXCLUDE_COLS = {TARGET, ID_COL, "kfold"}

# --- Cross Validation ---
N_FOLDS = 5
SEED = 42
RUN = 'v1'

# --- Feature Flags ---
FEATURE_FLAGS = {
    "ratios": False,
    "numerical_interactions": False,
    "categorical_interactions": False,
}

# --- LightGBM ---
LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "boosting": "gbdt",
    "data_sample_strategy": "goss",
    "num_iterations": 5000,
    "learning_rate": 0.01,
    "device_type": "cpu",
    "seed": SEED,
    "num_leaves": 31,
    "num_threads": -1,
    "max_depth": 6,
    "feature_fraction": 0.8,
    "early_stopping_round": 100,
    "lambda_l1": 0.1,
    "lambda_l2": 0.05,
    "verbosity": -1,
    "max_bin": 255,
    "metric": ["multi_logloss"],
}

# --- XGBoost ---
XGB_PARAMS = {
    "booster": "gbtree",
    "device": "cuda",
    "verbosity": 0,
    "validate_parameters": True,
    "num_boost_round": 5000,
    "early_stopping_rounds": 100,
    "eta": 0.01,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1,
    "alpha": 2,
    "tree_method": "hist",
    "grow_policy": "depthwise",
    "max_leaves": 31,
    "max_bin": 255,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "seed": SEED,
}

# --- CatBoost ---
CATBOOST_PARAMS = {
    "loss_function": "MultiClass",
    "eval_metric": "Accuracy",
    "iterations": 3000,
    "learning_rate": 0.01,
    "random_seed": SEED,
    "auto_class_weights": "Balanced",
    "l2_leaf_reg": 3.0,
    "depth": 6,
    "min_data_in_leaf": 1,
    "one_hot_max_size": 2,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
    "early_stopping_rounds": 50,
    "task_type": "GPU",
    "devices": "0",
    "verbose": 0,
}

# --- HistGBM ---
HISTGBM_PARAMS = {
    "loss": "log_loss",
    "learning_rate": 0.01,
    "max_iter": 3000,
    "max_leaf_nodes": 31,
    "max_depth": 8,
    "min_samples_leaf": 20,
    "l2_regularization": 0.5,
    "max_features": 0.8,
    "max_bins": 255,
    "early_stopping": True,
    "n_iter_no_change": 50,
    "validation_fraction": 0.1,
    "scoring": "accuracy",
    "verbose": 0,
    "random_state": SEED,
    "categorical_features": "from_dtype",
    "class_weight": "balanced",
}

# --- ExtraTrees ---
EXTRATREES_PARAMS = {
    "n_estimators": 1000,
    "criterion": "gini",
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": False,
    "n_jobs": -1,
    "random_state": SEED,
    "class_weight": "balanced"
}

# --- LogisticRegression ---
LOGISTIC_PARAMS = {
    "C": 0.5,
    "l1_ratio": 0.0,
    "class_weight": 'balanced',
    "random_state": SEED,
    "solver": "lbfgs",
    "max_iter": 2000,
    "verbose": 0
}

# --- XGB_TD --- 
XGB_TD_PARAMS = {
    'device': 'cuda',
    'random_state': SEED,
    'n_cv' : 1,
    'n_refit': 0,
    'n_repeats': 1,
    'n_threads': 12,
    'verbosity': 0,
    'val_metric_name': '1-balanced_accuracy',
    'val_fraction': 0.0
}

# --- LGBM_TD --- 
LGBM_TD_PARAMS = {
    'device': 'cpu',
    'random_state': SEED,
    'n_cv' : 1,
    'n_refit': 0,
    'n_repeats': 1,
    'n_threads': 12,
    'verbosity': 0,
    'val_metric_name': '1-balanced_accuracy',
    'val_fraction': 0.0
}

# --- CatBoost_TD --- 
CatBoost_TD_PARAMS = {
    'device': 'cpu',
    'random_state': SEED,
    'n_cv' : 1,
    'n_refit': 0,
    'n_repeats': 1,
    'n_threads': 12,
    'verbosity': 0,
    'val_fraction': 0.0
}

Resnet_RTDL_D_PARAMS = {
    "module_d_embedding": None,
    "module_d": 256,
    "module_d_hidden_factor": 2,
    "module_n_layers": 4,
    "verbose": 0,
    "max_epochs": 2,
    "batch_size": 256,
    "es_patience": 16,
    "device": 'cuda',
    "random_state": SEED,
    "n_cv": 1,
    "n_refit": 0,
    "n_repeats": 1,
    "n_threads": 12,
    "tmp_folder": 'tmp',
    "verbosity": 0,
    'val_fraction': 0.0
}