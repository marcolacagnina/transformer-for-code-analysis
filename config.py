import os
import torch

# --- Project Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# --- Data & Model I/O ---
# The pre-processed dataset is the main input for the project
DATASET_PATH = os.path.join(DATA_DIR, "final_dataset.json")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")
PLOT_SAVE_PATH = os.path.join(OUTPUT_DIR, "training_progress.png")

# --- Model & Tokenizer Config ---
TOKENIZER_NAME = "microsoft/codebert-base"
MAX_TOKEN_LENGTH = 512
CUSTOM_TAGS = [
    '<IMPORTS_START>', '<IMPORTS_END>',
    '<FUNC_DEF_START>', '<FUNC_DEF_END>',
    '<GLOBAL_CODE_START>', '<GLOBAL_CODE_END>'
]

# --- Model Hyperparameters ---
D_MODEL = 256
N_HEAD = 8
NUM_LAYERS = 4
DROPOUT = 0.4

# --- Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 10
LABEL_SMOOTHING = 0.1

# --- Early Stopping ---
ES_PATIENCE = 3
ES_DELTA = 1e-4

# --- Random Seed ---
SEED = 42