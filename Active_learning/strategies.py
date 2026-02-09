import numpy as np
import torch
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min

class StrategySelector:
    def __init__(self, agent, cfg):
        self.agent = agent
        self.cfg = cfg
        
    def select(self, X_pool, k, group_ids=None):
        """
        Main entry point for selection.
        Dispatches to specific logic based on config.QUERY_STRATEGY.
        Handles both point-wise and group-wise (Scenario) selection.
        """
        strategy = self.cfg.QUERY_STRATEGY
        
        # 0. 准备数据
        if isinstance(X_pool, np.ndarray):
            X_pool_tensor = torch.tensor(X_pool, dtype=torch.float32, device=self.agent.device)
        else:
            X_pool_tensor = X_pool

        # 1. Random Strategy
        if strategy == 'random':
            return self._select_random(X_pool, k, group_ids)
            
        # 2. Geometric / Diversity Strategies (Need Embeddings/Gradients)
        if strategy in ['badge', 'lcmd', 'coreset']:
            return self._select_geometric_strategy(X_pool_tensor, k, group_ids, strategy)
            
        # 3. Score-based Strategies (Uncertainty / Information Gain)
        # Includes: BALD, CIS, Entropy, Aleatoric
        scores = self._calculate_scores(X_pool_tensor, strategy)
        
        if group_ids is not None:
            return self._select_top_groups_by_score(scores, group_ids, k)
        else:
            return np.argsort(scores)[-k:]

    def _calculate_scores(self, X_tensor, strategy):
        # 基础预测
        mu_mean, epi_unc, alea_unc = self.agent.predict_with_uncertainties(X_tensor)
        epi = epi_unc.cpu().numpy().flatten()
        alea = alea_unc.cpu().numpy().flatten()
        
        def norm(x):
            _min, _max = np.min(x), np.max(x)
            if _max - _min < 1e-8: return np.zeros_like(x)
            return (x - _min) / (_max - _min)

        # --- BALD Strategy (Bayesian Active Learning by Disagreement) ---
        if strategy == 'bald':
            # Formula: H(y|x) - E_theta[H(y|x, theta)]
            # For Gaussian: Entropy = 0.5 * log(2*pi*e*sigma^2)
            
            # 1. Get raw ensemble predictions (N_ens, N_samples, 1)
            mus_stack, vars_stack = self.agent.predict_ensemble_raw(X_tensor)
            
            # 2. Total Uncertainty (Marginal Variance) = Aleatoric + Epistemic
            # epi_unc is Var(mu), alea_unc is Mean(var)
            total_var = epi_unc + alea_unc 
            entropy_total = 0.5 * torch.log(2 * np.pi * np.e * total_var + 1e-6)
            
            # 3. Expected Entropy (Mean of individual entropies)
            entropy_individual = 0.5 * torch.log(2 * np.pi * np.e * vars_stack + 1e-6)
            expected_entropy = entropy_individual.mean(dim=0).squeeze()
            
            # 4. Mutual Information
            bald_score = (entropy_total - expected_entropy).cpu().numpy().flatten()
            return bald_score

        # --- Existing Strategies ---
        elif strategy == 'cis':
            return norm(epi) - (self.cfg.BETA * norm(alea))
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
        # 获取 Embeddings (N, D)
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
        # 如果有 Scenario ID，我们必须对 Scenario 内的所有点求均值，得到 Scenario Vector
        if group_ids is not None:
            df = pd.DataFrame(feature_vectors)
            df['group'] = group_ids
            # 计算每个 Scenario 的中心向量
            group_means = df.groupby('group').mean()
            candidate_vectors = group_means.values
            candidate_ids = group_means.index.values
        else:
            candidate_vectors = feature_vectors
            candidate_ids = np.arange(len(feature_vectors))

        n_candidates = len(candidate_vectors)
        n_select = min(n_candidates, k)
        
        selected_indices_local = []
        
        # --- Strategy Logic ---
        
        if strategy == 'badge':
            # BADGE 使用 KMeans++ 初始化逻辑来选择中心
            # 1. 初始化
            kmeans = KMeans(n_clusters=n_select, init='k-means++', n_init=1, random_state=42)
            kmeans.fit(candidate_vectors)
            # 找出离 Cluster Centers 最近的真实样本点/组
            closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, candidate_vectors)
            selected_indices_local = closest_indices
            
        elif strategy == 'lcmd':
            # LCMD: 聚类后选最近的 (Representative Sampling)
            kmeans = KMeans(n_clusters=n_select, n_init=10, random_state=42)
            kmeans.fit(candidate_vectors)
            closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, candidate_vectors)
            selected_indices_local = closest_indices
            
        elif strategy == 'coreset':
            # k-Center Greedy (Farthest Point Sampling)
            # 1. 随机选第一个
            idx = np.random.randint(n_candidates)
            selected_indices_local = [idx]
            
            # 2. 维护最小距离矩阵
            dists = pairwise_distances(candidate_vectors[idx:idx+1], candidate_vectors).flatten()
            
            for _ in range(n_select - 1):
                # 选距离当前已选集最远的点
                new_idx = np.argmax(dists)
                selected_indices_local.append(new_idx)
                
                # 更新距离：新距离是 (旧距离) 和 (到新点的距离) 的最小值
                new_dists = pairwise_distances(candidate_vectors[new_idx:new_idx+1], candidate_vectors).flatten()
                dists = np.minimum(dists, new_dists)
        
        # --- Map back to original indices ---
        if group_ids is not None:
            selected_group_ids = candidate_ids[selected_indices_local]
            return np.where(np.isin(group_ids, selected_group_ids))[0]
        else:
            return candidate_ids[selected_indices_local]

    def _select_top_groups_by_score(self, scores, group_ids, k):
        df = pd.DataFrame({'score': scores, 'group': group_ids})
        # 计算每个组的平均分
        group_scores = df.groupby('group')['score'].mean()
        # 选分最高的 K 个组
        top_groups = group_scores.sort_values(ascending=False).head(k).index.values
        # 返回这些组对应的所有样本索引
        return np.where(np.isin(group_ids, top_groups))[0]

    def _select_random(self, X_pool, k, group_ids):
        if group_ids is not None:
            unique_groups = np.unique(group_ids)
            n_select = min(len(unique_groups), k)
            selected_groups = np.random.choice(unique_groups, size=n_select, replace=False)
            return np.where(np.isin(group_ids, selected_groups))[0]
        else:
            N = len(X_pool)
            return np.random.choice(N, size=min(N, k), replace=False)