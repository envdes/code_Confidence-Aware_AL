import numpy as np
import torch
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min

class StrategySelector:
    """
    Coordinates the data selection process. 
    Implements multiple Active Learning (AL) strategies to identify the most 
    informative samples from the unlabeled pool.
    """
    def __init__(self, agent, cfg):
        self.agent = agent
        self.cfg = cfg
        
    def select(self, X_pool, k, group_ids=None):
        """
        Main entry point for selection.

        Args:
            X_pool: The pool of unlabeled features (Tensor or Numpy).
            k: Number of samples (or groups) to select.
            group_ids: Optional array of IDs (e.g., Scenario_ID) to perform group-wise selection.
        """
        strategy = self.cfg.QUERY_STRATEGY
        
        if isinstance(X_pool, np.ndarray):
            X_pool_tensor = torch.tensor(X_pool, dtype=torch.float32, device=self.agent.device)
        else:
            X_pool_tensor = X_pool

        # Random Strategy: Baseline baseline for comparison
        if strategy == 'random':
            return self._select_random(X_pool, k, group_ids)
            
        # Geometric / Diversity Strategies (Need Embeddings/Gradients)
        if strategy in ['badge', 'lcmd', 'coreset']:
            return self._select_geometric_strategy(X_pool_tensor, k, group_ids, strategy)
            
        # Score-based Strategies (Uncertainty / Information Gain)
        # Includes: BALD, CIS, Entropy, Aleatoric
        scores = self._calculate_scores(X_pool_tensor, strategy)
        
        if group_ids is not None:
            return self._select_top_groups_by_score(scores, group_ids, k)
        else:
            return np.argsort(scores)[-k:]

    def _calculate_scores(self, X_tensor, strategy):
        mu_mean, epi_unc, alea_unc = self.agent.predict_with_uncertainties(X_tensor)
        epi = epi_unc.cpu().numpy().flatten()
        alea = alea_unc.cpu().numpy().flatten()
        
        def norm(x):
            """Min-Max normalization to bring scores into [0, 1] range."""
            _min, _max = np.min(x), np.max(x)
            if _max - _min < 1e-8: return np.zeros_like(x)
            return (x - _min) / (_max - _min)

        # --- BALD Strategy (Bayesian Active Learning by Disagreement) ---
        if strategy == 'bald':
            # Formula: H(y|x) - E_theta[H(y|x, theta)]
            # For Gaussian: Entropy = 0.5 * log(2*pi*e*sigma^2)
            
            # Get raw ensemble predictions (N_ens, N_samples, 1)
            mus_stack, vars_stack = self.agent.predict_ensemble_raw(X_tensor)
            
            # Total Uncertainty (Marginal Variance) = Aleatoric + Epistemic
            # epi_unc is Var(mu), alea_unc is Mean(var)
            total_var = epi_unc + alea_unc 
            entropy_total = 0.5 * torch.log(2 * np.pi * np.e * total_var + 1e-6)
            
            # 3. Expected Entropy (Mean of individual entropies)
            entropy_individual = 0.5 * torch.log(2 * np.pi * np.e * vars_stack + 1e-6)
            expected_entropy = entropy_individual.mean(dim=0).squeeze()
            
            # 4. Mutual Information
            bald_score = (entropy_total - expected_entropy).cpu().numpy().flatten()
            return bald_score

        # --- Information Source Strategies (Our Methods) --
        elif strategy == 'cis_gating':
            n_epi = norm(epi)
            n_alea = norm(alea)
            return np.power(n_epi + 1e-8, self.cfg.ALPHA) * np.power(1.0 - n_alea + 1e-8, self.cfg.BETA)
        elif strategy == 'entropy' or strategy == 'epistemic':
            return epi
        elif strategy == 'aleatoric':
            return alea
        else:
            raise ValueError(f"Unknown score-based strategy: {strategy}")

    def _select_geometric_strategy(self, X_tensor, k, group_ids, strategy):
        """Uses latent embeddings to select diverse/representative samples."""
        embeddings = self.agent.get_embeddings(X_tensor)
        
        feature_vectors = embeddings
        
        # --- BADGE Special Handling ---
        if strategy == 'badge':
            # BADGE Embedding = Gradient Embedding
            # For regression ensemble (LogitNormal), a robust proxy for gradient magnitude
            # is Embedding * sqrt(Epistemic_Uncertainty) or Embedding * sqrt(Total_Variance)
            _, epi_unc, _ = self.agent.predict_with_uncertainties(X_tensor)
            epi_std = np.sqrt(epi_unc.cpu().numpy()[:, np.newaxis])
            feature_vectors = embeddings * epi_std
            
        # --- Group Aggregation ---
        # If grouping is enabled, average embeddings across the scenario to create a "Scenario Vector"
        if group_ids is not None:
            df = pd.DataFrame(feature_vectors)
            df['group'] = group_ids
            group_means = df.groupby('group').mean()
            candidate_vectors = group_means.values
            candidate_ids = group_means.index.values
        else:
            candidate_vectors = feature_vectors
            candidate_ids = np.arange(len(feature_vectors))

        n_candidates = len(candidate_vectors)
        n_select = min(n_candidates, k)
        
        selected_indices_local = []
        
        # --- Geometric Sampling Logic ---
        if strategy == 'badge':
            # Uses KMeans++ initialization to find diverse, high-uncertainty centers
            kmeans = KMeans(n_clusters=n_select, init='k-means++', n_init=1, random_state=42)
            kmeans.fit(candidate_vectors)
            closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, candidate_vectors)
            selected_indices_local = closest_indices
            
        elif strategy == 'lcmd':
            # Large Cluster Maximum Distance: Standard KMeans to find representative centers
            kmeans = KMeans(n_clusters=n_select, n_init=10, random_state=42)
            kmeans.fit(candidate_vectors)
            closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, candidate_vectors)
            selected_indices_local = closest_indices
            
        elif strategy == 'coreset':
            # k-Center Greedy (Farthest Point Sampling): Iteratively selects the point# k-Center Greedy (Farthest Point Sampling): Iteratively selects the point
            idx = np.random.randint(n_candidates)
            selected_indices_local = [idx]
            dists = pairwise_distances(candidate_vectors[idx:idx+1], candidate_vectors).flatten()
            
            for _ in range(n_select - 1):
                new_idx = np.argmax(dists)
                selected_indices_local.append(new_idx)
                new_dists = pairwise_distances(candidate_vectors[new_idx:new_idx+1], candidate_vectors).flatten()
                dists = np.minimum(dists, new_dists)
        
        # --- Map back to original indices ---
        if group_ids is not None:
            selected_group_ids = candidate_ids[selected_indices_local]
            return np.where(np.isin(group_ids, selected_group_ids))[0]
        else:
            return candidate_ids[selected_indices_local]

    def _select_top_groups_by_score(self, scores, group_ids, k):
        # Score a group based on the average utility of all points within it
        df = pd.DataFrame({'score': scores, 'group': group_ids})
        # Score a group based on the average utility of all points within it
        group_scores = df.groupby('group')['score'].mean()
        top_groups = group_scores.sort_values(ascending=False).head(k).index.values
        return np.where(np.isin(group_ids, top_groups))[0]

    def _select_random(self, X_pool, k, group_ids):
        """Randomly selects samples or groups."""
        if group_ids is not None:
            unique_groups = np.unique(group_ids)
            n_select = min(len(unique_groups), k)
            selected_groups = np.random.choice(unique_groups, size=n_select, replace=False)
            return np.where(np.isin(group_ids, selected_groups))[0]
        else:
            N = len(X_pool)
            return np.random.choice(N, size=min(N, k), replace=False)