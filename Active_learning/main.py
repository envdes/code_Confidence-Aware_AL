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

# [修改] 导入重命名后的配置
import configs as cfg
from utils import *
from model import DeepEnsembleAgent, GaussianFTTransformer
from strategies import StrategySelector 

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

# ---------- Setup ----------
ensure_dir(cfg.RESULTS_DIR)
ensure_dir(cfg.MODEL_DIR)
ensure_dir(cfg.PLOT_DIR)
logger = init_logger(cfg.LOG_PATH)

logger.info("Start Active Learning run (Modular Version)")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
logger.info(f"Query Strategy: {cfg.QUERY_STRATEGY} | Loss: {cfg.LOSS_NAME}") 

def to_logit_space(y_tensor, eps=1e-6):
    y_clamped = torch.clamp(y_tensor, min=eps, max=1.0-eps)
    z = torch.log(y_clamped / (1.0 - y_clamped))
    return z

# ---------- Data Loading & Preprocessing ----------
features = [
    'O3 (ppb)', 'CO (ppb)', 'NO (ppb)', 'NOx (ppb)', 'ETH (ppb)', 'TOL(ppb)',
    'XYL (ppb)', 'ALD2 (ppb)', 'AONE (ppb)', 'PAR (ppb)', 'OLET (ppb)',
    'Temperature(K)', 'RH', 'BC (ug/m3)', 'OA (ug/m3)', 'NH4 (ug/m3)',
    'NO3 (ug/m3)', 'SO4 (ug/m3)'
]

logger.info("Loading data...")
# 假设数据路径相对位置不变
labeled_df = pd.read_csv('../PartMC_data/PartMC_labeled.csv')
unlabeled_df = pd.read_csv('../PartMC_data/PartMC_unlabeled.csv')
valid_df = pd.read_csv('../PartMC_data/PartMC_valid.csv')
test_df  = pd.read_csv('../PartMC_data/PartMC_test.csv')

scenarios_pool = unlabeled_df['Scenario_ID'].values
logger.info(f"Loaded Pool Scenarios. Unique Scenarios: {len(np.unique(scenarios_pool))}")

# Extract Values
X_train_np = labeled_df[features].values
y_train_np = labeled_df.iloc[:, 23].values

X_pool_np = unlabeled_df[features].values
y_pool_np = unlabeled_df.iloc[:, 23].values

X_valid_np = valid_df[features].values
y_valid_np = valid_df.iloc[:, 23].values
X_test_np  = test_df[features].values
y_test_np  = test_df.iloc[:, 23].values

# Scaling
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
# Evaluate Initial Model
val_preds, _ = learner.predict(X_valid)
val_preds = val_preds.cpu().numpy()
init_val_rmse = mean_squared_error(y_valid_orig_np, val_preds)
init_val_r2   = r2_score(y_valid_orig_np, val_preds)

test_preds, _ = learner.predict(X_test)
test_preds = test_preds.cpu().numpy()
init_test_rmse = mean_squared_error(y_test_orig_np, test_preds)
init_test_r2   = r2_score(y_test_orig_np, test_preds)

logger.info(f"Round 0 (Initial): Val RMSE={init_val_rmse:.4f}, R2={init_val_r2:.4f}")
logger.info(f"Round 0 (Initial): Test RMSE={init_test_rmse:.4f}, R2={init_test_r2:.4f}")

performance_log.append((0, init_val_rmse, init_val_r2, init_test_rmse, init_test_r2, 0))

# ---------- Active Learning Loop ----------
for i in range(cfg.N_QUERIES):
    logger.info(f"\n=== Query Round {i+1}/{cfg.N_QUERIES} ===")

    # Strategy Selection
    strategy_selector = StrategySelector(learner, cfg)
    
    n_inst = min(cfg.QUERY_BATCH_SIZE, X_pool_np.shape[0])
    logger.info(f"Selecting {n_inst} samples/groups using strategy: {cfg.QUERY_STRATEGY}...")
    
    # [关键] 调用 select，传入 scenarios_pool 以启用 Group Selection
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
    
    # Retrieve Data
    X_new = X_pool[query_idx]
    y_new_z = y_pool_z[query_idx]
    
    # Update Train
    X_train = torch.cat([X_train, X_new], dim=0)
    y_train_z = torch.cat([y_train_z, y_new_z], dim=0)
    
    # Remove from Pool
    mask = torch.ones(len(X_pool), dtype=torch.bool, device=device)
    mask[query_idx] = False
    X_pool = X_pool[mask]
    y_pool_z = y_pool_z[mask]
    y_pool = y_pool[mask]
    scenarios_pool = scenarios_pool[mask.cpu().numpy()]
    
    # Update Stats
    queried_indices_all.extend(query_idx.tolist())
    logger.info(f"Pool size: {len(X_pool)}, Train size: {len(X_train)}")
    
    # Retrain
    start_time = time.time()
    learner = get_new_learner()
    learner.fit(X_train, y_train_z, X_val=X_valid, y_val_z=y_valid_z)
    elapsed_time = time.time() - start_time
    logger.info(f"Training time: {elapsed_time:.2f}s")
    
    # Evaluate
    val_preds, _ = learner.predict(X_valid)
    val_preds = val_preds.cpu().numpy()
    rmse = mean_squared_error(y_valid_orig_np, val_preds)
    mae  = mean_absolute_error(y_valid_orig_np, val_preds)
    r2   = r2_score(y_valid_orig_np, val_preds)

    test_preds, _ = learner.predict(X_test)
    test_preds = test_preds.cpu().numpy()
    test_rmse = mean_squared_error(y_test_orig_np, test_preds)
    test_mae  = mean_absolute_error(y_test_orig_np, test_preds)
    test_r2   = r2_score(y_test_orig_np, test_preds)

    logger.info(f"Round {i+1}: Val RMSE={rmse:.4f}, R2={r2:.4f}")
    logger.info(f"Round {i+1}: Test RMSE={test_rmse:.4f}, R2={test_r2:.4f}")
    performance_log.append((i+1, rmse, mae, r2, test_rmse, test_mae, test_r2, elapsed_time))
        
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