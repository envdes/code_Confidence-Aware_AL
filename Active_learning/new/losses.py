import torch
import torch.nn as nn
import torch.nn.functional as F
import configs as cfg

class CAALLoss(nn.Module):
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
    
class FaithfulLoss(nn.Module):
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

    def forward(self, mu, var, target):
        loss_mse = F.mse_loss(mu, target)
        sq_error = (target - mu.detach())**2
        term1 = 0.5 * (sq_error / var)
        term2 = 0.5 * torch.log(var)
        loss_aleatoric = torch.mean(term1 + term2)
        return loss_mse + loss_aleatoric
    
class CAALNoDetachLoss(nn.Module):
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

    def forward(self, mu, var, target):
        loss_mse = F.mse_loss(mu, target)
        sq_error = (target - mu)**2
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


class NLL(nn.Module):
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

    def forward(self, mu, var, target):
        sq_error = (target - mu)**2
        term1 = 0.5 * (sq_error / var)
        term2 = 0.5 * torch.log(var)
        loss_aleatoric = torch.mean(term1 + term2)
        return loss_aleatoric

class NaturalGaussianNLLLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, natural_params, target):
        """
        natural_params: [Batch, 2] -> (eta1, eta2)
        target: [Batch, 1]
        """
        if target.ndim == 1:
            target = target.view(-1, 1)
        
        eta1 = natural_params[:, 0:1]
        eta2 = natural_params[:, 1:2]
        
        # 复现原代码逻辑：- (eta1*y + eta2*y^2 + eta1^2/(4*eta2) + 0.5*log(-2*eta2))
        term1 = eta1 * target + eta2 * (target ** 2)
        term2 = (eta1 ** 2) / (4 * eta2) + 0.5 * torch.log(-2 * eta2 + 1e-8)
        
        loss = -(term1 + term2)

        if self.reduction == 'mean':
            return loss.mean()
        return loss


def get_loss_function(name, **kwargs):
    loss_lam = float(cfg.LOSS_LAMBDA)
    loss_beta = float(cfg.LOSS_BETA)

    if name == 'mse_sgnll':
        return CAALLoss(weight=loss_lam)
    elif name == 'faithful':
        return FaithfulLoss(weight=loss_lam)
    elif name == 'mse_nll':
        return CAALNoDetachLoss(weight=loss_lam)
    elif name == 'beta_nll':
        return BetaNLLLoss(beta=loss_beta)
    elif name == 'nll_only':
        return NLL()
    elif name == 'nature_nll':
        return NaturalGaussianNLLLoss()
    else:
        raise ValueError(f"Unknown loss function name: {name}")