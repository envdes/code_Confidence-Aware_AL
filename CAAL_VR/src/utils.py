import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def init_logger(log_path):
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

def save_performance(perf_list, filepath):
    # Updated columns to match main.py
    columns = [
        'Step', 
        'Val_RMSE', 'Val_R2', 
        'Test_RMSE', 'Test_R2', 
        'Time_Sec'
    ]
    if not perf_list:
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(perf_list, columns=columns)
    df.to_csv(filepath, index=False)
    print(f"Performance saved to {filepath}")

def save_comprehensive_analysis(round_idx, learner, 
                                X_pool, y_pool_log, selected_idx, 
                                X_test, y_test_raw, 
                                save_dir):
    """
    Saves snapshot.
    y_pool_log: Passed as Log values (for distribution stats)
    y_test_raw: Passed as Real values (for error checking)
    """
    ensure_dir(save_dir)
    
    # --- Part A: POOL Analysis ---
    if isinstance(X_pool, np.ndarray):
        X_pool_tensor = torch.tensor(X_pool, dtype=torch.float32, device=learner.device)
    else:
        X_pool_tensor = X_pool

    # Pool Uncertainties (in Log Space)
    _, epi_pool, alea_pool = learner.predict_with_uncertainties(X_pool_tensor, mc_runs=30)
    epi_pool = epi_pool.cpu().numpy()
    alea_pool = alea_pool.cpu().numpy()
    
    # Pool Embeddings
    emb_pool = learner.get_embeddings(X_pool_tensor)
    
    # Selection Mask
    is_selected = np.zeros(len(epi_pool), dtype=int)
    is_selected[selected_idx] = 1
    
    # --- Part B: TEST Analysis ---
    if isinstance(X_test, np.ndarray):
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=learner.device)
    else:
        X_test_tensor = X_test

    # Test Predictions (Get raw log-space outputs for uncertainty)
    test_mu_log, epi_test, alea_test = learner.predict_with_uncertainties(X_test_tensor, mc_runs=30)
    
    test_mu_log = test_mu_log.cpu().numpy()
    # Convert Log Mean to Real Prediction
    test_preds_real = np.exp(test_mu_log)

    epi_test = epi_test.cpu().numpy()
    alea_test = alea_test.cpu().numpy()
    
    # --- Part C: Save ---
    save_path = os.path.join(save_dir, f'analysis_round_{round_idx:02d}.npz')
    
    np.savez(save_path, 
             pool_epi=epi_pool, 
             pool_alea=alea_pool, 
             pool_emb=emb_pool,
             pool_y_log=y_pool_log,
             is_selected=is_selected,
             test_preds_real=test_preds_real,
             test_y_real=y_test_raw,
             test_epi=epi_test,
             test_alea=alea_test,
             round_idx=round_idx)

    print(f"  [Analysis] Saved snapshot to {save_path}")