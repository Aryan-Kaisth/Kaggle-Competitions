# py
import os

# --- Paths ---
RAW_TRAIN = os.path.join("data", "raw", "train.csv")
RAW_TEST = os.path.join("data", "raw", "test.csv")

OOF_DIR = os.path.join("artifacts", "oof")
SUBMISSIONS_DIR = os.path.join("artifacts", "submissions")
TEST_PROBA_DIR = os.path.join("artifacts", "test_proba")

# --- Extras ---
TARGET = "irrigation_need"
TARGET_MAP = {"Low": 0, "Medium": 1, "High": 2}
TARGET_MAP_INV = {0: "Low", 1: "Medium", 2: "High"}
ID_COL = "id"

BASE_NUM_COLS = [
    'soil_ph',
    'soil_moisture',
    'organic_carbon',
    'electrical_conductivity',
    'temperature_c',
    'humidity',
    'rainfall_mm',
    'sunlight_hours',
    'wind_speed_kmh',
    'field_area_hectare',
    'previous_irrigation_mm']

BASE_CAT_COLS = [
    'soil_type',
    'crop_type',
    'crop_growth_stage',
    'season',
    'irrigation_type',
    'water_source',
    'mulching_used',
    'region']

# --- Cross Validation ---
N_FOLDS = 5
SEED = 42
SEED_LIST = [42, 71, 84, 69, 91]

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
    "metric": ["multi_logloss"]
}

# --- CatBoost ---
CATBOOST_PARAMS = {
    "loss_function": "MultiClass",
    "eval_metric": "TotalF1",
    "iterations": 5000,
    "learning_rate": 0.01,
    "random_seed": SEED,
    "auto_class_weights": "Balanced",
    "l2_leaf_reg": 2.0,
    "depth": 6,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
    "early_stopping_rounds": 200,
    "task_type": "GPU",
    "devices": "0",
    "verbose": 0,
    "use_best_model": True
}

# CatBoost ORDERED
CATBOOST_ORDERED_PARAMS = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "boosting_type": "Ordered",
    "iterations": 4000,
    "learning_rate": 0.05,
    "random_seed": SEED,
    "auto_class_weights": "Balanced",
    "l2_leaf_reg": 1.5,
    "depth": 4,
    "one_hot_max_size": 5,
    "bootstrap_type": "MVS",
    "early_stopping_rounds": 50,
    "random_strength": 1.0,
    "task_type": "CPU",
    "devices": "0",
    "verbose": 0,
    "use_best_model": True
}

# --- HistGBM ---
HISTGBM_PARAMS = {
    "loss": "log_loss",
    "learning_rate": 0.01,
    "max_iter": 1000,
    "max_leaf_nodes": 31,
    "max_depth": 6,
    "l2_regularization": 0.05,
    "max_features": 0.8,
    "early_stopping": False,
    "validation_fraction": None,
    "verbose": 0,
    "random_state": SEED,
    "categorical_features": "from_dtype",
    "class_weight": None
}

# --- ExtraTrees ---
EXTRATREES_PARAMS = {
    "n_estimators": 1000,
    "criterion": "gini",
    "max_depth": 8,
    "max_features": "sqrt",
    "bootstrap": False,
    "n_jobs": -1,
    "random_state": SEED,
    "verbose": 0,
    "class_weight": None
}

# --- LogisticRegression ---

LOGISTIC_PARAMS = {
    "class_weight": 'balanced',
    "random_state": SEED,
    "solver": "lbfgs",
    "max_iter": 10000,
    "verbose": 0
}

SGD_PARAMS = {
    "loss": "log_loss",
    "penalty": "l2",
    "alpha": 1e-5,
    "max_iter": 5000,
    "tol": 1e-4,
    "shuffle": True,
    "verbose": 0,
    "n_jobs": -1,
    "random_state": SEED,
    "learning_rate": "optimal",
    "early_stopping": False,
    "class_weight": 'balanced',
    "average": True
}

# --- XGB_TD --- 
XGB_TD_PARAMS = {
    'device': 'cuda',
    'random_state': SEED,
    'n_cv' : 1,
    'n_refit': 0,
    'n_repeats': 1,
    'val_fraction': 0.0,
    'n_threads': 12,
    'verbosity': 0,
    'train_metric_name': 'cross_entropy',
    'val_metric_name': '1-balanced_accuracy',
    'n_estimators': None,
    "max_depth": None,
    "lr": None,
    "subsample": None,
    "colsample_bytree": None,
    "colsample_bylevel": None,
    "colsample_bynode": None,
    "min_child_weight": None,
    "alpha": None,
    "reg_lambda": None,
    "gamma": None,
    "tree_method": None,
    "max_delta_step": None,
    "max_cat_to_onehot": None,
    "num_parallel_tree": None,
    "max_bin": None,
    "multi_strategy": None,
    "calibration_method": None
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

Resnet_RTDL_D_PARAMS = {
    "module_d_embedding": None,
    "module_d": 64,
    "module_d_hidden_factor": 2,
    "module_n_layers": 8,
    "verbose": 1,
    "max_epochs": 32,
    "batch_size": 512,
    "es_patience": 5,
    "use_checkpoints": True,
    "device": 'cuda',
    "random_state": SEED,
    "n_cv": 1,
    "n_refit": 0,
    "n_repeats": 1,
    "n_threads": 12,
    "tmp_folder": 'tmp',
    "verbosity": 1,
    'val_fraction': 0.0
}

RealMLP_TD_PARAMS = {
    "device": 'cuda',
    "random_state": SEED,
    "n_cv": 1,
    "n_refit": 0,
    "n_repeats": 1,
    "val_fraction": 0.0,
    "n_threads": 12,
    "verbosity": 0,
    "val_metric_name": '1-balanced_accuracy',
    "n_epochs": 5
}

TabM_D_PARAMS = {
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
    "arch_type": 'tabm-mini-normal',
    "tabm_k": 32,
    "num_emb_type": 'pwl',
    "num_emb_n_bins": 48,

    # training
    "batch_size": 512,
    "lr": None,
    "weight_decay": None,
    "n_epochs": 24,
    "patience": 16,

    # model size
    "d_embedding": None,
    "d_block": 128,
    "n_blocks": 6,
    "dropout": None,
    "compile_model": False,
    "allow_amp": True,
    "tfms": None,
    "gradient_clipping_norm": None,
    "share_training_batches": False,
    "val_metric_name": '1-balanced_accuracy',
    "train_metric_name": None
}