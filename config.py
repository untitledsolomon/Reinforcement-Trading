# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR   = os.path.join(BASE_DIR, "logs")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_EURUSD_1H_2020_2023.csv")
TEST_DATA_PATH  = os.path.join(DATA_DIR, "test_EURUSD_1H_2023_2025.csv")

# Environment defaults
DEFAULT_WINDOW_SIZE   = 30
DEFAULT_SL_OPTIONS    = [15, 30, 60]
DEFAULT_TP_OPTIONS    = [15, 30, 60]
DEFAULT_SPREAD_PIPS   = 1.0
DEFAULT_SLIPPAGE_PIPS = 0.2
DEFAULT_LOT_SIZE      = 100_000.0
INITIAL_EQUITY_USD    = 10_000.0
