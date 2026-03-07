# src/models/config.py
import os

# Paths
DATA_PATH = "data/raw/PS02_Training_set/PS02_Training_set.xlsx"
# DATA_PATH = "models/augmented_training.xlsx"
# DATA_PATH = "data/processed/training_data_augmented.xlsx"
EVIDENCE_DIR = "data/raw/PS02_Training_set/Evidences"
MODEL_DIR = "models"

# Ensure model dir exists
os.makedirs(MODEL_DIR, exist_ok=True)

# CSE Keywords for validation (optional)
CSE_KEYWORDS = ['nic', 'crsorgi', 'irctc', 'sbi', 'icici', 'hdfc', 'pnb', 'bob', 'airtel']