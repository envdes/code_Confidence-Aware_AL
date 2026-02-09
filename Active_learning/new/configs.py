import os

# ---------- Active Learning (AL) Configuration ----------
# Controls the iterative data selection and labeling process
SEED = 42                 # Random seed for reproducibility
N_ENSEMBLES = 6           # Number of models in the ensemble to estimate uncertainty
INIT_QUERY_SIZE = 100     # Number of samples in the initial labeled training set
N_QUERIES = 20            # Total number of active learning iterations/cycles
QUERY_BATCH_SIZE = 30     # Number of samples to select for labeling in each iteration
INPUT_FEATURES = 18       # Dimension of the input feature vector

# Selection Strategy for Active Learning
# Options: 
#   - Baselines: 'random' (baseline), 'ALM (entropy)', 'aleatoric' (data noise), 'confidence'. 'QBC (epi)'
#   - Ours: 'cis_gating' (CAAL)
#   - Advanced: 'bald' (Bayesian Disagreement), 'badge', 'lcmd', 'coreset'
QUERY_STRATEGY = 'cis_gating' 

# Acquisition Function Hyperparameters (Specific to CIS-Gating)
ALPHA = 1.0               # Weight for the epistemic uncertainty component. It is fixed, unless you want to use 'confidence' AL strategies with ALPHA=0.
BETA = 1.0                # Weight for the aleatoric/diversity component, BETA>=0

# ---------- Loss Function Configuration ----------
# Defines how the model handles uncertainty and prediction error
# Options: 'mse_sgnll' (ours), 'faithful', 'mse_nll', 'beta_nll', 'nll_only', 'natural_nll'
LOSS_NAME = 'mse_sgnll'

LOSS_LAMBDA = 0.1         # Weight for the aleatoric term in Logit-Normal Loss (for mse_sgnll and mse_nll)
LOSS_BETA = 0.5           # Beta parameter for robust NLL (Negative Log-Likelihood) (only for beta_nll)
DETACH_VAR_GRADIENT = False # If True, blocks gradient flow through the variance head (only for faithful)

# ---------- FT-Transformer Hyperparameters ----------
# Architectural settings for the Feature Tokenizer + Transformer model
D_MODEL = 128             # Embedding dimension for each feature
N_HEAD = 8                # Number of attention heads
NUM_LAYERS = 4            # Number of Transformer encoder blocks
DIM_FEEDFORWARD = 512     # Dimension of the hidden layer in the MLP block
DROPOUT = 0.0             # Probability of dropout for regularization

# ---------- Training Hyperparameters ----------
LEARNING_RATE = 1e-4      # Initial optimizer learning rate
WEIGHT_DECAY = 1e-4       # L2 regularization coefficient
BATCH_SIZE = 128          # Samples per training batch
AL_EPOCHS = 100           # Max training epochs per active learning cycle

# ---------- Optimization & Early Stopping ----------
EARLY_STOP_PATIENCE = 20  # Stop training if validation loss doesn't improve for X epochs
EARLY_STOP_MIN_DELTA = 1e-4 # Minimum change to qualify as an improvement
LR_PATIENCE = 10          # Epochs to wait before reducing learning rate
LR_FACTOR = 0.5           # Multiplier to reduce LR (LR = LR * factor)
LR_MIN = 1e-7             # Lower bound for learning rate reduction

# ---------- File System & Logging Paths ----------
RESULTS_DIR = './results_modular'
MODEL_DIR = os.path.join(RESULTS_DIR, 'models')        # Saved model weights (.pt)
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')          # Training curves and metrics
ANALYSIS_DIR = os.path.join(RESULTS_DIR, 'analysis_data') # Intermediate data for research
LOG_PATH = os.path.join(RESULTS_DIR, 'log.txt')        # Text logs of the training process
PERFORMANCE_PATH = os.path.join(RESULTS_DIR, 'performance.csv') # Metrics per AL cycle
QUERIED_IDX_PATH = os.path.join(RESULTS_DIR, 'queried_indices.npy') # History of labeled samples