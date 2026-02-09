import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

def ensure_dir(path):
    """Creates the directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def init_logger(log_path):
    """Initializes the system logger."""
    # Clear previous handlers to prevent duplicate logs in notebooks or loops
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = logging.getLogger()
        
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s')

    # File Handler (writes logs to log.txt)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler (prints logs to console/terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def save_model(model, path):
    """
    Save a PyTorch model's state_dict to the given path.
    """
    torch.save(model.state_dict(), path)

def save_performance(perf_list, filepath):
    """
    Saves the performance metrics list to a CSV file.
    
    NOTE: The column names must match the number of values appended in main.py.
    Current configuration supports 8 columns:
    Step, Val_RMSE, Val_MAE, Val_R2, Test_RMSE, Test_MAE, Test_R2, Time_Sec
    """
    columns = [
        'Step', 
        'Val_RMSE', 'Val_MAE', 'Val_R2', 
        'Test_RMSE', 'Test_MAE', 'Test_R2', 
        'Time_Sec'
    ]
    
    # Create a DataFrame (handle empty list case)
    if not perf_list:
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(perf_list, columns=columns)
        
    df.to_csv(filepath, index=False)
    print(f"Performance saved to {filepath}")
def save_plot(values, path, title='Metric Over Time', ylabel='Value'):
    """
    Plot a line chart of the given values and save to file.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(values, marker='o')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Query Iteration')
    plt.grid(True)
    plt.savefig(path)
    plt.close()
    
# ------------------------------------------------------------------------------
# ANALYSIS & VISUALIZATION STRATEGY (Rationale for Data Saving)
# ------------------------------------------------------------------------------
# Since our Active Learning strategy selects data at the "Scenario Level" (groups)
# but the model trains at the "Point Level", we save raw data to support a 
# Two-Level Comparative Analysis in the paper:
#
# 1. Macro View (Scenario-Level) - "The Decision Logic"
#    - How to plot: Aggregating scores by `pool_scenario_ids` (Mean Epi vs. Mean Alea).
#    - Goal: Demonstrate that the CIS strategy effectively targets "High-Value Scenarios"
#      (High Information + Low Noise). This validates the selection algorithm itself.
#
# 2. Micro View (Point-Level) - "The Purity Check"
#    - How to plot: Scatter plot of individual points (Raw Epi vs. Raw Alea).
#    - Goal: Defend against the critique: "Does selecting by group introduce hidden noise?"
#      We prove that even at the granular level, the points inside selected scenarios 
#      remain clean (high SNR), preventing "poisonous" data from entering the training set.
#
# 3. Validation (Test Set) - "The Justification"
#    - Goal: Correlate `test_alea` with `test_errors` to prove that avoiding 
#      high-aleatoric data is theoretically sound (Noise = Unlearnable Error).
# ------------------------------------------------------------------------------
    
def save_comprehensive_analysis(round_idx, learner, 
                                X_pool, y_pool_orig, selected_idx, pool_scenario_ids, 
                                X_test, y_test_orig, 
                                save_dir):
    """
    Saves a comprehensive snapshot for Deep Dive Analysis.
    
    Data saved to .npz:
    1. Pool Context: 
       - Uncertainties (Epi/Alea) -> For Signal vs Noise analysis
       - Embeddings (CLS token)   -> For Diversity/Clustering analysis (t-SNE)
       - True Labels (y_pool)     -> For Oracle/Difficulty analysis (High Error vs High Uncertainty)
    2. Selection Logic: 
       - is_selected mask         -> To visualize which points were picked
    3. Test Set Dynamics: 
       - Test Preds & Uncertainties -> For Calibration Analysis (Does model know what it doesn't know?)
    """
    ensure_dir(save_dir)
    
    # --- Part A: POOL Analysis (Why did we pick these?) ---
    if isinstance(X_pool, np.ndarray):
        X_pool_tensor = torch.tensor(X_pool, dtype=torch.float32, device=learner.device)
    else:
        X_pool_tensor = X_pool

    # 1. Pool Uncertainties (MC Dropout / Ensemble stats)
    # Using 30 runs for speed, increase to 50 for higher precision if needed
    _, epi_pool, alea_pool = learner.predict_with_uncertainties(X_pool_tensor, mc_runs=30)
    epi_pool = epi_pool.cpu().numpy()
    alea_pool = alea_pool.cpu().numpy()
    
    # 2. Pool Embeddings (CRITICAL for visualizing diversity)
    # This allows you to run t-SNE later to see if selected points are clustered or spread out
    emb_pool = learner.get_embeddings(X_pool_tensor)
    
    # 3. Selection Mask
    is_selected = np.zeros(len(epi_pool), dtype=int)
    is_selected[selected_idx] = 1
    
    # --- Part B: TEST Analysis (Did the model actually improve?) ---
    if isinstance(X_test, np.ndarray):
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=learner.device)
    else:
        X_test_tensor = X_test

    # 4. Test Predictions & Uncertainties
    # Essential for "Calibration Curve" (Error vs Uncertainty)
    test_preds, epi_test, alea_test = learner.predict_with_uncertainties(X_test_tensor, mc_runs=30)
    test_preds = test_preds.cpu().numpy()
    epi_test = epi_test.cpu().numpy()
    alea_test = alea_test.cpu().numpy()
    
    # --- Part C: Save All to Compressed Numpy File ---
    save_path = os.path.join(save_dir, f'analysis_round_{round_idx:02d}.npz')
    
    np.savez(save_path, 
             # Pool Data
             pool_epi=epi_pool, 
             pool_alea=alea_pool, 
             pool_emb=emb_pool,           # [New] Latent features
             pool_y=y_pool_orig,          # [New] Ground truth for "Oracle" check
             pool_scenarios=pool_scenario_ids,
             is_selected=is_selected,
             
             # Test Data
             test_preds=test_preds,       # [New] Current model predictions
             test_y=y_test_orig,          # [New] Ground truth
             test_epi=epi_test,           # [New] Test uncertainty
             test_alea=alea_test,
             
             round_idx=round_idx)

    print(f"  [Analysis] Saved snapshot to {save_path}")