import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
# Import modular configuration and custom source modules
import configs as cfg
from src.utils import *
from src.model import DeepEnsembleAgent, GaussianFTTransformer
from src.strategies import StrategySelector 

def parse_arguments():
    parser = argparse.ArgumentParser(description="Active Learning Experiment Runner")

    # --- System & GPU ---
    parser.add_argument('--seed', type=int, default=cfg.SEED, 
                        help='Random seed for reproducibility.')

    # --- Active Learning Strategy ---
    parser.add_argument('--strategy', type=str, default=cfg.QUERY_STRATEGY, 
                        help="Strategy name (e.g., 'cis_gating', 'random', 'Ale', 'Entropy', 'BLAD', 'LCMD', 'BADGE', 'Coreset').")
    parser.add_argument('--n_queries', type=int, default=cfg.N_QUERIES, 
                        help='Number of active learning rounds.')
    parser.add_argument('--query_batch_size', type=int, default=cfg.QUERY_BATCH_SIZE, 
                        help='Number of samples to query per round.')
    
    # --- Loss Function & Hyperparameters ---
    parser.add_argument('--loss', type=str, default=cfg.LOSS_NAME, 
                        help="Loss function (e.g., 'mse_sgnll', 'faithful', 'mse_nll', 'beta_nll', 'nll_only', 'nature_nll').")
    parser.add_argument('--loss_lambda', type=float, default=cfg.LOSS_LAMBDA, 
                        help='Weight for the aleatoric uncertainty term (Lambda). only work for mse_sgnll and mse_nll')
    parser.add_argument('--loss_beta', type=float, default=cfg.LOSS_BETA, 
                        help='Beta hyperparameter for Beta-NLL loss.')
    parser.add_argument('--detach_grad', action='store_true', 
                        help='If set, detach variance gradient (for stability). True for faithful')

    # --- 4. Strategy Specific (CIS / CIS-Gating) ---
    parser.add_argument('--alpha', type=float, default=cfg.ALPHA, 
                        help='Alpha parameter for CIS-Gating (Epistemic weight). Only use for Confidence strategy with Alpha=0')
    parser.add_argument('--beta', type=float, default=cfg.BETA, 
                        help='Beta parameter for CIS-Gating (Aleatoric penalty).')
    
    # --- 5. Training Hyperparameters ---
    parser.add_argument('--lr', type=float, default=cfg.LEARNING_RATE, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=cfg.AL_EPOCHS, help='Training epochs per round.')
    parser.add_argument('--n_ensembles', type=int, default=cfg.N_ENSEMBLES, help='Number of models in the ensemble.')

    return parser.parse_args()

args = parse_arguments()

print(f"\n{'='*40}")
print(f"  CONFIGURATION UPDATE")
print(f"{'='*40}")
# System
cfg.SEED = args.seed
print(f"[*] System: Seed={cfg.SEED}")

# AL Settings
cfg.QUERY_STRATEGY = args.strategy
cfg.N_QUERIES = args.n_queries
cfg.QUERY_BATCH_SIZE = args.query_batch_size
cfg.ALPHA = args.alpha
cfg.BETA = args.beta
print(f"[*] Strategy: {cfg.QUERY_STRATEGY}")
print(f"    - Alpha: {cfg.ALPHA}, Beta: {cfg.BETA}")
print(f"    - Rounds: {cfg.N_QUERIES}, Batch: {cfg.QUERY_BATCH_SIZE}")

# Loss Settings
cfg.LOSS_NAME = args.loss
cfg.LOSS_LAMBDA = args.loss_lambda
cfg.LOSS_BETA = args.loss_beta
cfg.DETACH_VAR_GRADIENT = args.detach_grad
print(f"[*] Loss: {cfg.LOSS_NAME}")
print(f"    - Lambda: {cfg.LOSS_LAMBDA}, Loss Beta: {cfg.LOSS_BETA}")

# Training
cfg.LEARNING_RATE = args.lr
cfg.BATCH_SIZE = args.batch_size
cfg.AL_EPOCHS = args.epochs
cfg.N_ENSEMBLES = args.n_ensembles
print(f"[*] Train: LR={cfg.LEARNING_RATE}, BS={cfg.BATCH_SIZE}, Epochs={cfg.AL_EPOCHS}, Ensembles={cfg.N_ENSEMBLES}")
print(f"{'='*40}\n")

# 1. Determine Experiment Group (Folder Category)
if args.strategy in ['random', 'entropy', 'bald', 'badge', 'coreset']:
    exp_group = "01_Baselines"
elif args.strategy in ['cis', 'cis_gating']:
    # Check if this is a standard run or an ablation study
    is_standard_params = (args.alpha == 1.0 and args.beta == 1.0 and args.loss_lambda == 0.1)
    
    if is_standard_params:
        exp_group = "02_Proposed_Method"
    else:
        exp_group = "03_Ablation_Studies"
else:
    exp_group = "99_Misc_Experiments"

# 2. Generate Unique Run Name based on parameters
# Format: {Strategy}_{Loss}_a{Alpha}_b{Beta}_lam{Lambda}_seed{Seed}
run_name = f"{args.strategy}_{args.loss}"

# Only add Alpha/Beta to the name if the strategy uses them
if args.strategy in ['cis', 'cis_gating']:
    run_name += f"_a{args.alpha}_b{args.beta}"

# Only add Lambda if it's not the default 0.1 (keeps names shorter)
if args.loss_lambda != 0.1:
    run_name += f"_lam{args.loss_lambda}"

run_name += f"_seed{args.seed}"

# 3. Construct the Full Path
# Example: ./experiments/02_Proposed_Method/cis_gating_faithful_a1.0_b1.0_seed42/
cfg.RESULTS_DIR = os.path.join("experiments", exp_group, run_name)

# 4. Update Sub-directories based on new RESULTS_DIR
cfg.MODEL_DIR = os.path.join(cfg.RESULTS_DIR, 'models')
cfg.PLOT_DIR = os.path.join(cfg.RESULTS_DIR, 'plots')
cfg.ANALYSIS_DIR = os.path.join(cfg.RESULTS_DIR, 'analysis_data')
cfg.LOG_PATH = os.path.join(cfg.RESULTS_DIR, 'log.txt')
cfg.PERFORMANCE_PATH = os.path.join(cfg.RESULTS_DIR, 'performance.csv')
cfg.QUERIED_IDX_PATH = os.path.join(cfg.RESULTS_DIR, 'queried_indices.npy')

# 5. Create Directories
ensure_dir(cfg.RESULTS_DIR)
ensure_dir(cfg.MODEL_DIR)
ensure_dir(cfg.PLOT_DIR)
ensure_dir(cfg.ANALYSIS_DIR)

print(f"[*] Results will be saved to: {cfg.RESULTS_DIR}")

# ---------- Reproducibility ----------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():                  
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(cfg.SEED)

# ---------- Setup & Logging ----------
ensure_dir(cfg.RESULTS_DIR)
ensure_dir(cfg.MODEL_DIR)
ensure_dir(cfg.PLOT_DIR)
logger = init_logger(cfg.LOG_PATH)

logger.info("Start Active Learning run (Modular Version)")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
logger.info(f"Query Strategy: {cfg.QUERY_STRATEGY} | Loss: {cfg.LOSS_NAME}") 

def to_logit_space(y_tensor, eps=1e-6):
    """
    Transforms target values from [0, 1] to logit space (-inf, +inf).
    Required for Gaussian modeling of bounded outputs.
    """
    y_clamped = torch.clamp(y_tensor, min=eps, max=1.0-eps)
    z = torch.log(y_clamped / (1.0 - y_clamped))
    return z

# ---------- Data Loading & Preprocessing ----------
# Features for PartMC simulation dataset
features = [
    'O3 (ppb)', 'CO (ppb)', 'NO (ppb)', 'NOx (ppb)', 'ETH (ppb)', 'TOL(ppb)',
    'XYL (ppb)', 'ALD2 (ppb)', 'AONE (ppb)', 'PAR (ppb)', 'OLET (ppb)',
    'Temperature(K)', 'RH', 'BC (ug/m3)', 'OA (ug/m3)', 'NH4 (ug/m3)',
    'NO3 (ug/m3)', 'SO4 (ug/m3)'
]


logger.info("Loading data...")
labeled_df = pd.read_csv('../PartMC_data/PartMC_labeled.csv')
unlabeled_df = pd.read_csv('../PartMC_data/PartMC_unlabeled.csv')
valid_df = pd.read_csv('../PartMC_data/PartMC_valid.csv')
test_df  = pd.read_csv('../PartMC_data/PartMC_test.csv')

# scenario_pool identifies which physical scenario each data point belongs to (for Group Selection)
scenarios_pool = unlabeled_df['Scenario_ID'].values
logger.info(f"Loaded Pool Scenarios. Unique Scenarios: {len(np.unique(scenarios_pool))}")

X_train_np = labeled_df[features].values
y_train_np = labeled_df['Chi'].values

X_pool_np = unlabeled_df[features].values
y_pool_np = unlabeled_df['Chi'].values

X_valid_np = valid_df[features].values
y_valid_np = valid_df['Chi'].values
X_test_np  = test_df[features].values
y_test_np  = test_df['Chi'].values

# Standardize features based on the initial training set
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train_np)
X_pool_np  = scaler.transform(X_pool_np)
X_valid_np = scaler.transform(X_valid_np)
X_test_np  = scaler.transform(X_test_np)

# Convert to Tensors
X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train_np, dtype=torch.float32, device=device)
X_pool = torch.tensor(X_pool_np, dtype=torch.float32, device=device)
y_pool = torch.tensor(y_pool_np, dtype=torch.float32, device=device)
X_valid = torch.tensor(X_valid_np, dtype=torch.float32, device=device)
y_valid = torch.tensor(y_valid_np, dtype=torch.float32, device=device)
X_test  = torch.tensor(X_test_np,  dtype=torch.float32, device=device)
y_test  = torch.tensor(y_test_np,  dtype=torch.float32, device=device)

y_valid_orig_np = y_valid.cpu().numpy()
y_test_orig_np  = y_test.cpu().numpy()

# Logit Transform
y_train_z = to_logit_space(y_train)
y_pool_z  = to_logit_space(y_pool) 
y_valid_z = to_logit_space(y_valid)

# Learner Builder
def get_new_learner():
    """Initializes a fresh Deep Ensemble of FT-Transformers."""
    model_params = {
        'in_features': cfg.INPUT_FEATURES,
        'd_model': cfg.D_MODEL,
        'nhead': cfg.N_HEAD,
        'num_layers': cfg.NUM_LAYERS,
        'dim_feedforward': cfg.DIM_FEEDFORWARD,
        'dropout_p': cfg.DROPOUT
    }
    
    return DeepEnsembleAgent(
        model_class=GaussianFTTransformer,
        model_params=model_params,
        lr=cfg.LEARNING_RATE, 
        weight_decay=cfg.WEIGHT_DECAY,
        epochs=cfg.AL_EPOCHS, 
        batch_size=cfg.BATCH_SIZE,
        n_ensembles=cfg.N_ENSEMBLES 
    )

# ---------- Initial Training (Round 0) ----------
logger.info(f"Start Initial Training on {len(X_train)} samples...")
learner = get_new_learner()
learner.fit(X_train, y_train_z, X_val=X_valid, y_val_z=y_valid_z)

performance_log = []
queried_indices_all = []
total_start_time = time.time()

logger.info("Evaluating Initial Model (Round 0)...")
# Predict and evaluate on the validation and test sets
val_preds, _ = learner.predict(X_valid)
val_preds = val_preds.cpu().numpy()
init_val_mse = mean_squared_error(y_valid_orig_np, val_preds)
init_val_r2   = r2_score(y_valid_orig_np, val_preds)

test_preds, _ = learner.predict(X_test)
test_preds = test_preds.cpu().numpy()
init_test_mse = mean_squared_error(y_test_orig_np, test_preds)
init_test_r2   = r2_score(y_test_orig_np, test_preds)

logger.info(f"Round 0 (Initial): Val MSE={init_val_mse:.4f}, R2={init_val_r2:.4f}")
logger.info(f"Round 0 (Initial): Test MSE={init_test_mse:.4f}, R2={init_test_r2:.4f}")

performance_log.append((0, init_val_mse, init_val_r2, init_test_mse, init_test_r2, 0))

# ---------- Active Learning Loop ----------
for i in range(cfg.N_QUERIES):
    logger.info(f"\n=== Query Round {i+1}/{cfg.N_QUERIES} ===")

    # Initialize strategy selector with current model state
    strategy_selector = StrategySelector(learner, cfg)
    
    n_inst = min(cfg.QUERY_BATCH_SIZE, X_pool_np.shape[0])
    logger.info(f"Selecting {n_inst} samples/groups using strategy: {cfg.QUERY_STRATEGY}...")
    
    # Select high-utility samples from the pool based on the strategy
    query_idx = strategy_selector.select(
        X_pool, 
        k=n_inst, 
        group_ids=scenarios_pool 
    )
    
    selected_ids = np.unique(scenarios_pool[query_idx])
    logger.info(f"Selected Scenario IDs: {selected_ids}")
    logger.info(f"Total points selected: {len(query_idx)}")
    
    current_y_pool_np = y_pool.cpu().numpy()
    
    # Save Analysis Snapshot
    save_comprehensive_analysis(
        round_idx=i+1,
        learner=learner,
        X_pool=X_pool,
        y_pool_orig=current_y_pool_np,
        selected_idx=query_idx,
        pool_scenario_ids=scenarios_pool,
        X_test=X_test,
        y_test_orig=y_test_orig_np,
        save_dir=cfg.ANALYSIS_DIR
    )
    
    X_new = X_pool[query_idx]
    y_new_z = y_pool_z[query_idx]
    
    X_train = torch.cat([X_train, X_new], dim=0)
    y_train_z = torch.cat([y_train_z, y_new_z], dim=0)
    
    mask = torch.ones(len(X_pool), dtype=torch.bool, device=device)
    mask[query_idx] = False
    X_pool = X_pool[mask]
    y_pool_z = y_pool_z[mask]
    y_pool = y_pool[mask]
    scenarios_pool = scenarios_pool[mask.cpu().numpy()]
    
    queried_indices_all.extend(query_idx.tolist())
    logger.info(f"Pool size: {len(X_pool)}, Train size: {len(X_train)}")
    
    start_time = time.time()
    learner = get_new_learner()
    learner.fit(X_train, y_train_z, X_val=X_valid, y_val_z=y_valid_z)
    elapsed_time = time.time() - start_time
    logger.info(f"Training time: {elapsed_time:.2f}s")
    
    # Evaluate
    val_preds, _ = learner.predict(X_valid)
    val_preds = val_preds.cpu().numpy()
    mse = mean_squared_error(y_valid_orig_np, val_preds)
    mae  = mean_absolute_error(y_valid_orig_np, val_preds)
    r2   = r2_score(y_valid_orig_np, val_preds)

    test_preds, _ = learner.predict(X_test)
    test_preds = test_preds.cpu().numpy()
    test_mse = mean_squared_error(y_test_orig_np, test_preds)
    test_mae  = mean_absolute_error(y_test_orig_np, test_preds)
    test_r2   = r2_score(y_test_orig_np, test_preds)

    logger.info(f"Round {i+1}: Val MSE={mse:.4f}, R2={r2:.4f}")
    logger.info(f"Round {i+1}: Test MSE={test_mse:.4f}, R2={test_r2:.4f}")
    performance_log.append((i+1, mse, mae, r2, test_mse, test_mae, test_r2, elapsed_time))
        
    # Save Models
    for idx, model in enumerate(learner.models):
        torch.save(
            model.state_dict(),
            f"{cfg.MODEL_DIR}/model_round_{i+1}_ensemble_{idx}.pt"
        )

# ---------- Final Wrap-up ----------
total_elapsed_time = time.time() - total_start_time
logger.info(f"Total Active Learning time: {total_elapsed_time:.2f}s")

np.save(cfg.QUERIED_IDX_PATH, np.array(queried_indices_all))
save_performance(performance_log, cfg.PERFORMANCE_PATH)

logger.info("Finished Active Learning Run.")