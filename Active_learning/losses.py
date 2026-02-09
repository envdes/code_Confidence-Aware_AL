import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitNormalLoss(nn.Module):
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

    def forward(self, mu, var, target):
        loss_mse = F.mse_loss(mu, target)
        sq_error = (target - mu.detach())**2
        term1 = 0.5 * (sq_error / var)
        term2 = 0.5 * torch.log(var)
        loss_aleatoric = torch.mean(term1 + term2)
        return loss_mse + self.weight * loss_aleatoric

class BetaNLLLoss(nn.Module):
    def __init__(self, beta=0.5):
        super().__init__()
        self.beta = beta

    def forward(self, mu, var, target):
        sq_error = (target - mu)**2
        nll_per = 0.5 * (torch.log(var) + sq_error / var)
        beta_weight = var.detach() ** self.beta
        return torch.mean(beta_weight * nll_per)

class MSEOnlyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, var, target):
        return F.mse_loss(mu, target)

def get_loss_function(name, **kwargs):
    if name == 'logit_normal':
        return LogitNormalLoss(weight=kwargs.get('aleatoric_weight', 0.1))
    elif name == 'beta_nll':
        return BetaNLLLoss(beta=kwargs.get('beta', 0.5))
    elif name == 'mse_only':
        return MSEOnlyLoss()
    else:
        raise ValueError(f"Unknown loss function name: {name}")