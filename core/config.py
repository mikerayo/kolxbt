"""
Configuration for KOL Tracker ML System
"""

import os
from pathlib import Path

# Base paths - adjusted for new structure (core/ is a subdirectory)
BASE_DIR = Path(__file__).parent.parent  # Go up to project root
DATA_DIR = BASE_DIR / "data"  # Data folder is at project root
DB_DIR = DATA_DIR  # Database in data folder

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)

# Input data file
KOLS_DATA_FILE = BASE_DIR / "kolscan_complete_kols_all_periods.json"

# Database
DATABASE_URL = f"sqlite:///{DB_DIR / 'kol_tracker.db'}"

# Solana RPC Configuration
# Using Helius RPC (faster, higher rate limits)
SOLANA_RPC_URL = "https://mainnet.helius-rpc.com/?api-key=6ed9747d-d26e-4a19-bc2d-d4cc44d00b11"

# Alternative RPCs (commented out, uncomment if needed)
# SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"
# SOLANA_RPC_URL = "https://solana-api.projectserum.com"
# SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", SOLANA_RPC_URL)

# RPC Request Settings
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30  # seconds
RATE_LIMIT_DELAY = 2.0  # seconds between requests (increased for Helius rate limits)

# DEX Program IDs
DEX_PROGRAMS = {
    # Major DEXs
    "raydium": "675kPX9MHTjS2zt1qf1WQiJrNJ4UcQz7fQvVq9D7iB",
    "jupiter": "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",
    "orca": "9W959DqEETiGZocYGoQkVcjyMVfUcRqNYsY2Vm5vXojH",

    # MEMECOIN DEX - CRITICAL for 100x gems
    # pump.fun / pumpswap - Where most 100x memes start
    "pump_fun": "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P",
}

# WSOL Mint Address (Wrapped SOL)
WSOL_MINT = "So11111111111111111111111111111111111111112"

# Trading Criteria
MIN_HOLD_TIME_SECONDS = 5 * 60  # 5 minutes
TARGET_MULTIPLE = 3.0  # 3x

# Tracking Settings
HISTORY_DAYS = 30  # How many days of history to fetch
MAX_SIGNATURES_PER_REQUEST = 1000  # Max signatures per RPC call

# ML Model Settings
DIAMOND_HAND_WEIGHTS = {
    'three_x_plus_rate': 0.35,
    'win_rate': 0.25,
    'avg_hold_time': 0.20,
    'consistency_score': 0.15,
    'total_trades': 0.05,
}

SCALPER_HOLD_TIME_THRESHOLD = 5 * 60  # 5 minutes
DIAMOND_HAND_MIN_HOLD_TIME = 5 * 60  # 5 minutes
DIAMOND_HAND_MIN_3X_RATE = 0.20  # At least 20% of trades are 3x+

# Cluster Settings (K-Means)
N_CLUSTERS = 5
CLUSTER_NAMES = [
    "Diamond Hands",
    "Scalpers",
    "Losers",
    "Inconsistent",
    "New Traders"
]

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = DATA_DIR / "kol_tracker.log"

# Report Settings
REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# DEEP LEARNING / PYTORCH CONFIGURATION
# =============================================================================

# Model Directories
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Model Registry
MODEL_REGISTRY_PATH = MODELS_DIR / "model_registry.json"
TRAINER_STATE_PATH = MODELS_DIR / "trainer_state.json"

# Training Configuration
TOKEN_PREDICTOR_CONFIG = {
    'input_dim': 7,  # Number of features
    'hidden_dims': [64, 32],  # Simplified architecture
    'dropout': 0.2,
}

KOL_PREDICTOR_CONFIG = {
    'input_dim': 7,  # Number of features
    'embed_dim': 256,
    'num_heads': 4,
    'hidden_dims': [128, 64],
    'dropout': 0.4,
    'use_attention': True,
}

# Training Hyperparameters
TRAINING_CONFIG = {
    # Token Predictor
    'token_epochs': 50,
    'token_batch_size': 32,
    'token_learning_rate': 0.001,
    'token_weight_decay': 1e-5,
    'token_early_stopping_patience': 10,

    # KOL Predictor
    'kol_epochs': 50,
    'kol_batch_size': 32,
    'kol_learning_rate': 0.001,
    'kol_weight_decay': 1e-5,
    'kol_early_stopping_patience': 10,
}

# Continuous Training Settings
CONTINUOUS_TRAINING_CONFIG = {
    'retrain_interval_hours': 6,  # Retrain every 6 hours
    'check_interval_seconds': 300,  # Check every 5 minutes
    'min_data_age_hours': 24,  # Minimum data age before retraining
    'min_samples_for_training': 100,  # Minimum samples required
}

# Data Configuration
DATA_CONFIG = {
    # Token Predictor
    'token_history_days': 60,  # Days of historical data for training
    'token_train_split': 0.8,  # Train/validation split ratio
    'token_min_trades_per_kol': 3,  # Minimum trades required per KOL

    # KOL Predictor
    'kol_history_days': 90,  # Days of historical features
    'kol_prediction_offset_days': 35,  # Days in past to predict from
    'kol_min_trades_per_kol': 5,  # Minimum trades required
}

# Model Versioning
MODEL_VERSIONING = {
    'keep_n_best_versions': 5,  # Keep only N best versions
    'prune_old_versions': True,  # Automatically prune old versions
}

# Target Metrics (for model evaluation)
TARGET_METRICS = {
    # Token Predictor
    'token_min_auc': 0.75,
    'token_target_auc': 0.80,
    'token_min_precision': 0.70,
    'token_min_recall': 0.60,

    # KOL Predictor
    'kol_max_rmse': 0.15,
    'kol_target_rmse': 0.10,
    'kol_min_r2': 0.60,
    'kol_max_mae': 0.10,
}

# Device Configuration
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_NUM_WORKERS = 0  # DataLoader workers (0 for Windows compatibility)
