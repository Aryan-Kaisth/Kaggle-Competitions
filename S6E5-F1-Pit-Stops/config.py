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
    "boosting_type": "gbdt",
    'data_sample_strategy': 'goss',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 256,
    'min_data_in_leaf':40,
    'max_depth': -1,
    'learning_rate': 0.01, # Lower Lr for stable training
    'n_estimators': 10_000,
    'class_weight': 'balanced',
    'random_state': SEED,
    'n_jobs': -1,
    'importance_type': 'gain',
    'colsample_bytree': 0.5,
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

# --- Resnet ---
Resnet_PARAMS = {
    "module_d_embedding": None,
    "module_d": 256,
    "module_d_hidden_factor": 2,
    "module_n_layers": 6,
    "verbose": 1,
    "max_epochs": 32,
    "batch_size": 1024,
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

# --- RealMLP ---
REALMLP_PARAMS = {
    'random_state': SEED,
    'verbosity': 2,
    'val_metric_name': '1-auc_ovr',

    'n_ens': 24,
    'n_epochs': 6,
    'batch_size': 256,
    'use_early_stopping': False,
    'early_stopping_additive_patience': 10,
    'early_stopping_multiplicative_patience': 1,

    'lr': 0.01,
    'wd': 0.016,
    'sq_mom': 0.99,
    'lr_sched': 'lin_cos_log_15',
    'first_layer_lr_factor': 0.25,

    'embedding_size': 6,
    'max_one_hot_cat_size': 18,
    'hidden_sizes': [512, 256, 128],
    'act': 'silu',
    'p_drop': 0.05,
    'p_drop_sched': 'invsqrtp1e-3',

    'plr_hidden_1': 16,
    'plr_hidden_2': 8,
    'plr_act_name': 'gelu',
    'plr_lr_factor': 0.1151,
    'plr_sigma': 2.33,

    'ls_eps': 0.01,
    'ls_eps_sched': 'sqrt_cos',

    'add_front_scale': False,
    'bias_init_mode': 'neg-uniform-dynamic-2',
    'tfms': ['one_hot', 'median_center', 'robust_scale',
             'smooth_clip', 'embedding', 'l2_normalize'],
}

CATGBM_PARAMS = {
    'iterations':10_000,
    'learning_rate':0.01,
    'depth': 6,
    'l2_leaf_reg':8.0,
    'random_strength':0.8,
    'bootstrap_type':"Bayesian",
    'grow_policy': 'SymmetricTree',
    'bagging_temperature':0.35,
    'loss_function':"Logloss",
    'eval_metric':"AUC",
    'task_type':"GPU",
    'random_seed':SEED,
    'early_stopping_rounds':100,
    'use_best_model': True,
    'border_count': 255,
    "auto_class_weights": "Balanced"
}

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
    "num_emb_n_bins": 48,

    # training
    "batch_size": 1028,
    "lr": None,
    "weight_decay": None,
    "n_epochs": 5,
    "patience": 5,

    # model size
    "d_embedding": None,
    "d_block": 256,
    "n_blocks": 6,
    "dropout": None,
    "compile_model": False,
    "allow_amp": True,
    "tfms": None,
    "gradient_clipping_norm": None,
    "share_training_batches": False,
    "val_metric_name": '1-auc_ovr',
    "train_metric_name": None
}

# FTT
FTT_PARAMS = {
    "module_d_token": 4,
    "module_d_ffn_factor": None,
    "module_n_layers": 6,
    "module_n_heads": None,
    "module_token_bias": None,
    "module_attention_dropout": None,
    "module_ffn_dropout": None,
    "module_residual_dropout": None,
    "module_activation": None,
    "module_prenormalization": True,
    "module_initialization": None,
    "module_kv_compression": None,
    "module_kv_compression_sharing": None,
    "verbose": None,
    "max_epochs": 8,
    "batch_size": 512,
    "optimizer": None,
    "optimizer_weight_decay": None,
    "es_patience": None,
    "lr": None,
    "lr_scheduler": None,
    "lr_patience": None,
    "use_checkpoints": True,
    "transformed_target": None,
    "tfms": None,
    "quantile_output_distribution": None,
    "val_metric_name": None,
    "device": 'cuda',
    "random_state": SEED,
    "n_cv": 1,
    "n_refit": 0,
    "n_repeats": 1,
    "val_fraction": 0.0,
    "n_threads": None,
    "tmp_folder": None,
    "verbosity": 0,
    "calibration_method": None
}