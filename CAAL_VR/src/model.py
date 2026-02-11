import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import configs as cfg
from src.losses import get_loss_function

# ---------- FT-Transformer Architecture ----------
class GaussianFTTransformer(nn.Module):
    """
    Feature Tokenizer Transformer tailored for Gaussian probabilistic regression.
    It transforms tabular features into embeddings and outputs both Mean and Variance.
    """
    def __init__(self, in_features=18, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout_p=0.0):
        super().__init__()
        self.d_model = d_model
        self.feature_weights = nn.Parameter(torch.randn(in_features, d_model))
        self.feature_bias = nn.Parameter(torch.randn(in_features, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout_p, batch_first=True, activation='gelu', norm_first=True
        )        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.out_mean = nn.Linear(d_model, 1)    
        self.out_raw_var = nn.Linear(d_model, 1) 
        
    def forward(self, x, return_embedding=False, detach_var_from_feature=False, use_natural_params=False): 
        batch_size = x.shape[0]
        # Feature Tokenization: [batch, features] -> [batch, features, d_model]
        x_expanded = x.unsqueeze(-1)
        x_emb = x_expanded * self.feature_weights + self.feature_bias

        # Prepend CLS token to the feature sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_input = torch.cat((cls_tokens, x_emb), dim=1) 

        x_out = self.transformer_encoder(x_input)
        cls_out = self.norm(x_out[:, 0, :])

        if use_natural_params:
            # Predict natural parameters (eta1, eta2) for Natural Gaussian NL
            eta1 = self.out_mean(cls_out)
            raw_eta2 = self.out_raw_var(cls_out)
            eta2 = -F.softplus(raw_eta2) - 1e-6
            output = torch.cat([eta1, eta2], dim=1)
        else:
            # Standard Parameterization: Mean and Variance
            mu_z = self.out_mean(cls_out)
            cls_for_var = cls_out.detach() if detach_var_from_feature else cls_out
            var_z = F.softplus(self.out_raw_var(cls_for_var)) + 1e-6
            output = (mu_z, var_z)

        if return_embedding:
            return (*output, cls_out) if use_natural_params else (*output, cls_out)
        return output

# ---------- Deep Ensemble Agent ----------
class DeepEnsembleAgent:
    """
    Manages an ensemble of models to estimate Epistemic and Aleatoric uncertainty.
    """
    def __init__(self, model_class, model_params, lr, weight_decay, epochs, batch_size, n_ensembles=5):
        self.model_class = model_class
        self.model_params = model_params
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_ensembles = n_ensembles
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []
        
        # Load the loss function specified in config
        self.loss_fn = get_loss_function(cfg.LOSS_NAME)
        
    def fit(self, X, y_z, X_val=None, y_val_z=None):
        """Trains N independent models on the same (or bootstrapped) data."""
        self.models = []
        
        if isinstance(X, np.ndarray): X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if isinstance(y_z, np.ndarray): y_z = torch.tensor(y_z, dtype=torch.float32, device=self.device)
        X, y_z = X.to(self.device), y_z.to(self.device)
        
        has_val = (X_val is not None) and (y_val_z is not None)
        if has_val:
            if isinstance(X_val, np.ndarray): X_val = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            if isinstance(y_val_z, np.ndarray): y_val_z = torch.tensor(y_val_z, dtype=torch.float32, device=self.device)
            X_val, y_val_z = X_val.to(self.device), y_val_z.to(self.device)

        train_ds = torch.utils.data.TensorDataset(X, y_z)

        if cfg.LOSS_NAME == 'faithful':
            strict_detach_mode = True
        else:
            strict_detach_mode = getattr(cfg, 'DETACH_VAR_GRADIENT', False)
        
        use_natural = (cfg.LOSS_NAME == 'nature_nll')
        
        for i in range(self.n_ensembles):
            model = self.model_class(**self.model_params).to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            
            best_loss = float('inf')
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
            for epoch in range(self.epochs):
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    yb = yb.view(-1, 1)

                    if use_natural:
                        nat_params = model(xb, use_natural_params=True)
                        loss = self.loss_fn(nat_params, yb)
                    else:
                        mu_z, var_z = model(xb, detach_var_from_feature=strict_detach_mode)
                        loss = self.loss_fn(mu_z, var_z, yb)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                if has_val:
                    model.eval()
                    with torch.no_grad():
                        v_mu, _ = model(X_val)
                        val_loss = F.mse_loss(v_mu, y_val_z.view(-1, 1)).item()
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_state = copy.deepcopy(model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter > 10: break 
                        
            if has_val: model.load_state_dict(best_state)
            self.models.append(model)                            

    def predict(self, X):
        """Returns EXP(mu) to give predictions in the original scale (Real World Values)."""
        mu_z, _, _ = self._predict_ensemble_logit(X)
        preds = torch.exp(mu_z).cpu().view(-1)
        return preds, torch.zeros_like(preds)
        
    def predict_with_uncertainties(self, X, mc_runs=None):
        """Aggregates ensemble outputs into Mean, Epistemic (disagreement), and Aleatoric (noise)."""
        mu_mean_z, epi_z, alea_z = self._predict_ensemble_logit(X)
        return mu_mean_z, epi_z, alea_z
        
    def _predict_ensemble_logit(self, X):
        """Internal helper to compute ensemble stats in logit space."""
        if isinstance(X, np.ndarray): X = torch.tensor(X, dtype=torch.float32, device=self.device)
        X = X.to(self.device)
        use_natural = (cfg.LOSS_NAME == 'nature_nll')
        mus, vars = [], []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                if use_natural:
                    p = model(X, use_natural_params=True)
                    m = -p[:, 0:1] / (2 * p[:, 1:2]) # mu = -eta1 / 2*eta2
                    v = -1.0 / (2 * p[:, 1:2])       # var = -1 / 2*eta2
                else:
                    m, v = model(X)
                mus.append(m)
                vars.append(v)
        mus_stack = torch.stack(mus, dim=0)
        vars_stack = torch.stack(vars, dim=0)
        mu_mean = mus_stack.mean(dim=0).squeeze()
        epi_unc = mus_stack.var(dim=0).squeeze()
        alea_unc = vars_stack.mean(dim=0).squeeze()
        return mu_mean, epi_unc, alea_unc

    def predict_ensemble_raw(self, X):
        """Returns the raw (mu, var) pairs for every individual ensemble member for BALD."""
        if isinstance(X, np.ndarray): X = torch.tensor(X, dtype=torch.float32, device=self.device)
        X = X.to(self.device)
        mus, vars = [], []
        use_natural = (cfg.LOSS_NAME == 'nature_nll')
        with torch.no_grad():
            for model in self.models:
                model.eval()
                if use_natural:
                    p = model(X, use_natural_params=True)
                    m, v = -p[:, 0:1] / (2 * p[:, 1:2]), -1.0 / (2 * p[:, 1:2])
                else:
                    m, v = model(X)
                mus.append(m)
                vars.append(v)
        # return shapes: (n_ensemble, n_samples, 1)
        return torch.stack(mus, dim=0), torch.stack(vars, dim=0)

    def get_embeddings(self, X):
        """Extracts the average CLS-token embedding across the ensemble for LCMD, BDAGE and Coreset."""
        if isinstance(X, np.ndarray): X = torch.tensor(X, dtype=torch.float32, device=self.device)
        X = X.to(self.device)
        embeddings_list = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                _, _, emb = model(X, return_embedding=True)
                embeddings_list.append(emb)
        avg_embedding = torch.stack(embeddings_list, dim=0).mean(dim=0)
        return avg_embedding.cpu().numpy()