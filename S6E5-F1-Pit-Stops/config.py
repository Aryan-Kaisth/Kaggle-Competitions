from pathlib import Path

# --- Paths ---
TRAIN_PATH = Path("data", "raw", "train.csv")
TEST_PATH = Path("data", "raw", "test.csv")
ORIG_PATH = Path("data", "orig", "f1_strategy_dataset_v4.csv")

OOF_PROBA_DIR = Path("artifacts", "oof")
SUBMISSIONS_DIR = Path("artifacts", "submissions")
TEST_PROBA_DIR = Path("artifacts", "test_proba")

# --- Features ---
TARGET = "pitnextlap"
ID_COL = "id"

BASE_CAT_COLS = ['driver', 'compound', 'race']

BASE_NUM_COLS = [
    'year', 
    'lapnumber', 
    'stint', 
    'tyrelife', 
    'position', 
    'laptime (s)', 
    'laptime_delta', 
    'cumulative_degradation', 
    'raceprogress', 
    'position_change',
    'pitstop'
]

N_FOLDS = 5
SEED = 42
SEED_LIST = [42, 71, 84, 69, 91]

# --- Logistic Regression ---
LR_PARAMS = {
    "class_weight": 'balanced',
    "random_state": SEED,
    "solver": "lbfgs",
    "max_iter": 5000,
    "verbose": 0
}

# --- HistGBM ---
HISTGBM_PARAMS = {
    "loss": "log_loss",
    "learning_rate": 0.01,
    "max_iter": 5000,
    "max_leaf_nodes": 31,
    "max_depth": 5,
    "l2_regularization": 0.05,
    "max_features": 0.8,
    "early_stopping": False,
    "validation_fraction": None,
    "verbose": 0,
    "random_state": SEED,
    "class_weight": 'balanced'
}

# --- LightGBM ---
LGBM_PARAMS = {
    'boosting_type': 'goss',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'max_depth': -1,
    'learning_rate': 0.01, # Lower Lr for stable training
    'n_estimators': 10_000,
    'class_weight': 'balanced',
    'random_state': SEED,
    'n_jobs': -1,
    'importance_type': 'gain',
    'colsample_bytree': 0.8,
    'verbose': -1
}

# --- XGBoost ---
XGBM_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'device': 'cuda',
    'learning_rate': 0.01, # Lower Lr for stable training
    'n_estimators': 10_000,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.2,
    'reg_lambda': 1,
    'gamma': 0.1,
    'random_state': SEED,
    'n_jobs': -1,
    'verbosity': 0,
    'class_weight': 'balanced',
    'early_stopping_rounds': 200 
}