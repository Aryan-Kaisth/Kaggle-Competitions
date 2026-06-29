from pathlib import Path

TRAIN_DATA_PATH = Path(r"data\raw\train.csv")
TEST_DATA_PATH = Path(r"data\raw\test.csv")
ORIG_DATA_PATH = Path(r"data\orig\star_classification.csv")

ID_COL = "id"
TARGET_COL = "class"
SEED = 42
N_FOLDS = 5
N_CLASSES = 3

OOF_PROBA_DIR = Path(r"artifacts\oof_proba")
TEST_PROBA_DIR = Path(r"artifacts\test_proba")
SUBMISSION_DIR = Path(r"artifacts\submissions")

# Logistic Regression
LR_PARAMS = {
    "class_weight": "balanced",
    "random_state": SEED,
    "solver": "lbfgs",
    "max_iter": 5000,
    "verbose": 0,
}

# Histgbm
HISTGBM_PARAMS = {
    "loss": "log_loss",
    "learning_rate": 0.03,
    "max_iter": 2000,
    "max_leaf_nodes": 31,
    "max_features": 0.8,
    "early_stopping": False,
    "validation_fraction": None,
    "verbose": 0,
    "random_state": SEED,
    "class_weight": "balanced",
    "categorical_features": "from_dtype",
}

# LightGBM
LGBM_PARAMS = {
    "boosting_type": "gbdt",
    'data_sample_strategy': 'goss',
    'objective': 'multiclass',
    'num_classes': N_CLASSES,
    "metric": "multi_logloss",
    "num_leaves": 31,
    'max_depth': -1,
    'learning_rate': 0.03,
    'n_estimators': 5_000,
    'class_weight': 'balanced',
    'random_state': SEED,
    'n_jobs': -1,
    'importance_type': 'gain',
    'colsample_bytree': 0.8,
    'verbose': -1
}

# XGBoost
XGB_PARAMS = {
    'objective': 'multi:softprob',
    "num_class": N_CLASSES,
    'eval_metric': 'mlogloss',
    'tree_method': 'hist',
    'device': 'cuda',
    'learning_rate': 0.01,
    'n_estimators': 5000,
    'max_depth': 3,
    'subsample': 0.75,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.8,
    'random_state': SEED,
    'n_jobs': -1,
    'verbosity': 0,
    'class_weight': 'balanced',
    'early_stopping_rounds': 150,
    "enable_categorical": True
}

# ResNet
RESNET_PARAMS = {
    "module_d": 256,
    "module_d_hidden_factor": 2,
    "module_n_layers": 4,
    "verbose": 1,
    "max_epochs": 16,
    "batch_size": 1024,
    "es_patience": 10,
    "use_checkpoints": True,
    "device": 'cuda',
    "random_state": SEED,
    "n_cv": 1,
    "n_refit": 0,
    "n_repeats": 1,
    "n_threads": 12,
    "tmp_folder": 'tmp',
    "val_metric_name": "cross_entropy",
    "verbosity": 1,
    'val_fraction': 0.0
}

# --- CatBoost ---
CATBOOST_PARAMS = {
    "loss_function": "MultiClass",
    "eval_metric": "TotalF1",
    "iterations": 4000,
    "learning_rate": 0.03,
    "random_seed": SEED,
    "auto_class_weights": "Balanced",
    "bootstrap_type": "Bayesian",
    "early_stopping_rounds": 100,
    "task_type": "CPU",
    "verbose": 0,
    "use_best_model": True,
    "boosting_type": "Ordered"
}

# TABM
TABM_PARAMS = {
    "device": 'cuda',
    "random_state": SEED,
    "n_cv": 1,
    "n_refit": 0,
    "n_repeats": 1,
    "val_fraction": 0.0,
    "n_threads": 12,
    "tmp_folder": None,
    "verbosity": 1,

    # architecture
    "arch_type": 'tabm',
    "num_emb_type": 'pwl',
    "num_emb_n_bins": None,

    # training
    "batch_size": 256,
    "lr": None,
    "weight_decay": None,
    "n_epochs": 32,
    "patience": 5,

    # model size
    "d_embedding": None,
    "d_block": 1024,
    "n_blocks": 6,
    "dropout": True,
    "compile_model": False,
    "allow_amp": True,
    "tfms": None,
    "gradient_clipping_norm": True,
    "share_training_batches": True,
    "val_metric_name": '1-auc_ovr',
    "train_metric_name": None
}