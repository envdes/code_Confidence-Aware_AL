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

# ---------- Data Loading & Preprocessing ----------
# Features for PartMC simulation dataset
FEATURES_LIST = [
    'Temperature(K)', 'RH', 'CO (ppb)', 'NOx (ppb)', 'O3 (ppb)', 'SO2_av',
    'BC (ug/m3)', 'PM2.5 (ug/m3)'
]

logger.info("Loading data...")
labeled_df = pd.read_csv('../VR_data/ERL_labelled.csv')
unlabeled_df = pd.read_csv('../VR_data/ERL_unlabelled.csv')
valid_df = pd.read_csv('../VR_data/ERL_val.csv')
test_df  = pd.read_csv('../VR_data/ERL_test.csv')

# Standardize features based on the initial training set
scaler = StandardScaler()
X_train_np = scaler.fit_transform(labeled_df[FEATURES_LIST].values)
X_pool_np  = scaler.transform(unlabeled_df[FEATURES_LIST].values)
X_val_np   = scaler.transform(valid_df[FEATURES_LIST].values)
X_test_np  = scaler.transform(test_df[FEATURES_LIST].values)

# Log Transform Targets
y_train_z = torch.tensor(np.log(labeled_df['VR'].values), dtype=torch.float32, device=device)
y_pool_z  = torch.tensor(np.log(unlabeled_df['VR'].values), dtype=torch.float32, device=device)
y_val_z   = torch.tensor(np.log(valid_df['VR'].values), dtype=torch.float32, device=device)

# Keep raw 'y' for Oracle/Evaluation (Test Set doesn't need Log for training, just for comparison)
y_pool_raw_np = unlabeled_df['VR'].values
y_val_raw_np  = valid_df['VR'].values
y_test_raw_np = test_df['VR'].values

X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
X_pool  = torch.tensor(X_pool_np, dtype=torch.float32, device=device)
X_val   = torch.tensor(X_val_np, dtype=torch.float32, device=device)
X_test  = torch.tensor(X_test_np, dtype=torch.float32, device=device)

logger.info(f"Data Loaded: Train={len(X_train)}, Pool={len(X_pool)}, Val={len(X_val)}, Test={len(X_test)}")

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

# Evaluation Helper
def evaluate_real_world(agent, X_tensor, y_raw_numpy):
    # Agent.predict returns EXP(mu) because of our change in model.py
    preds_real, _ = agent.predict(X_tensor) 
    preds_real = preds_real.cpu().numpy()
    
    # Calculate metrics on the Real Scale
    rmse = np.sqrt(mean_squared_error(y_raw_numpy, preds_real))
    r2 = r2_score(y_raw_numpy, preds_real)
    return rmse, r2

# ---------- Initial Training (Round 0) ----------
logger.info(f"Start Initial Training on {len(X_train)} samples...")
learner = get_new_learner()
learner.fit(X_train, y_train_z, X_val=X_val, y_val_z=y_val_z)

performance_log = []
queried_indices_all = []
total_start_time = time.time()

logger.info("Evaluating Initial Model (Round 0)...")
# Predict and evaluate on the validation and test sets
init_val_rmse, init_val_r2 = evaluate_real_world(learner, X_val, y_val_raw_np)
init_test_rmse, init_test_r2 = evaluate_real_world(learner, X_test, y_test_raw_np)

logger.info(f"Round 0 (Initial): Val MSE={init_val_rmse:.4f}, R2={init_val_r2:.4f}")
logger.info(f"Round 0 (Initial): Test MSE={init_test_rmse:.4f}, R2={init_test_r2:.4f}")

performance_log.append((0, init_val_rmse, init_val_r2, init_test_rmse, init_test_r2, 0))

# ---------- Active Learning Loop ----------
for i in range(cfg.N_QUERIES):
    logger.info(f"\n=== Query Round {i+1}/{cfg.N_QUERIES} ===")

    # Initialize strategy selector with current model state
    strategy_selector = StrategySelector(learner, cfg)
    
    n_inst = min(cfg.QUERY_BATCH_SIZE, X_pool_np.shape[0])
    logger.info(f"Selecting {n_inst} samples/group using strategy: {cfg.QUERY_STRATEGY}...")
    
    # Select Indices (Point-based)
    query_idx = strategy_selector.select(
        X_pool, 
        k=n_inst
    )
    
    
    # Save Analysis Snapshot
    save_comprehensive_analysis(
        round_idx=i+1,
        learner=learner,
        X_pool=X_pool,
        y_pool_log=y_pool_z.cpu().numpy(),
        selected_idx=query_idx,
        X_test=X_test,
        y_test_raw=y_test_raw_np,
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
    y_pool_raw_np = y_pool_raw_np[mask.cpu().numpy()]
    
    queried_indices_all.extend(query_idx.tolist())
    logger.info(f"Pool size: {len(X_pool)}, Train size: {len(X_train)}")
    
    start_time = time.time()
    learner = get_new_learner()
    learner.fit(X_train, y_train_z, X_val=X_val, y_val_z=y_val_z)
    elapsed_time = time.time() - start_time
    logger.info(f"Training time: {elapsed_time:.2f}s")
    
    # Evaluate
    val_rmse, val_r2 = evaluate_real_world(learner, X_val, y_val_raw_np)
    test_rmse, test_r2 = evaluate_real_world(learner, X_test, y_test_raw_np)

    logger.info(f"Round {i+1}: Val RMSE={val_rmse:.4f}, R2={val_r2:.4f}")
    logger.info(f"Round {i+1}: Test RMSE={test_rmse:.4f}, R2={test_r2:.4f}")
    performance_log.append((i+1, val_rmse, val_r2, test_rmse, test_r2, elapsed_time))
        
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