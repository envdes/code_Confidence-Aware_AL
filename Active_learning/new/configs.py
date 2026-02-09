import os

# ---------- Active Learning Settings ----------
SEED = 42
N_ENSEMBLES = 6
INIT_QUERY_SIZE = 100
N_QUERIES = 20
QUERY_BATCH_SIZE = 30
INPUT_FEATURES = 18

# [关键修改] 策略选择
# Options: 
#   Baselines: 'random', 'entropy', 'aleatoric'
#   Ours: 'cis', 'cis_gating'
#   New Additions: 'bald', 'badge', 'lcmd', 'coreset'
QUERY_STRATEGY = 'cis_gating' 

# Acquisition Function Hyperparameters (for CIS/CIS-Gating)
ALPHA = 1.0
BETA = 1.0 

# [关键修改] Loss设置
# Options: 'mse_sgnll', 'faithful','mse_nll', 'beta_nll', 'nll_only', 'natural_nll'
LOSS_NAME = 'faithful'

LOSS_LAMBDA = 0.1  # Logit-Normal Loss 中 aleatoric 项的权重
LOSS_BETA = 0.5
DETACH_VAR_GRADIENT = False
# ---------- FT-Transformer Hyperparameters ----------
D_MODEL = 128        
N_HEAD = 8           
NUM_LAYERS = 4       
DIM_FEEDFORWARD = 512
DROPOUT = 0.0

# ---------- Training Hyperparameters ----------
LEARNING_RATE = 1e-4 
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
AL_EPOCHS = 100      

# ---------- Training Tricks ----------
EARLY_STOP_PATIENCE = 20
EARLY_STOP_MIN_DELTA = 1e-4
LR_PATIENCE = 10
LR_FACTOR = 0.5
LR_MIN = 1e-7

# ---------- Paths ----------
RESULTS_DIR = './results_modular'
MODEL_DIR = os.path.join(RESULTS_DIR, 'models')
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')
ANALYSIS_DIR = os.path.join(RESULTS_DIR, 'analysis_data') 
LOG_PATH = os.path.join(RESULTS_DIR, 'log.txt')
PERFORMANCE_PATH = os.path.join(RESULTS_DIR, 'performance.csv')
QUERIED_IDX_PATH = os.path.join(RESULTS_DIR, 'queried_indices.npy')