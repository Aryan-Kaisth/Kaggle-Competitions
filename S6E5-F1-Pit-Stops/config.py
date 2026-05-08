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

N_FOLDS = 5
SEED = 42
SEED_LIST = [42, 71, 84, 69, 91]