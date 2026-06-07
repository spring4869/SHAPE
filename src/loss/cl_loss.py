import torch
import torch.nn as nn
import torch.nn.functional as F

class LHInfoNCELoss(nn.Module):
    """
    Strict Lorentzian–Euclidean Hybrid InfoNCE Loss
    (Numerically Stable Version)
    """
    def __init__(
        self,
        embedding_dim,
        dim_lorentz=None,
        beta=1.0,
        gamma_power=0.5, 
        temperature=0.2,
        eps=1e-6,
    ):
        super().__init__()
        self.beta = beta
        self.gamma_power = gamma_power
        self.temperature = temperature
        self.eps = eps

        if dim_lorentz is None:
            self.dim_lorentz = embedding_dim // 2
        else:
            self.dim_lorentz = dim_lorentz

        self.dim_euclid = embedding_dim - self.dim_lorentz

    def gamma(self, r):
        """γ(r) = r^p"""
        return torch.pow(r + self.eps, self.gamma_power)

    def to_lorentz(self, z):
        """
        Cosh Projection: Euclidean -> Hyperboloid
        """
        r = torch.norm(z, dim=-1, keepdim=True).clamp(min=self.eps)
        r_tilde = self.gamma(r)
        r_tilde = r_tilde.clamp(max=15.0) 

        direction = z / r

        # x0^2 - x_rest^2 = beta
        sqrt_beta = self.beta ** 0.5
        x0 = sqrt_beta * torch.cosh(r_tilde)
        x_rest = sqrt_beta * torch.sinh(r_tilde) * direction

        return torch.cat([x0, x_rest], dim=-1)

    def lorentz_similarity(self, z):
        z_l = self.to_lorentz(z) # (2B, D+1)

        # (2B, 1, D+1) * (1, 2B, D+1) -> (2B, 2B) 
        x = z_l.unsqueeze(1)
        y = z_l.unsqueeze(0)
        
        # <x, y> = -x0*y0 + x_rest*y_rest
        inner = -x[..., 0] * y[..., 0] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
        
        # d^2 ≈ 2 * (|inner| - beta)
        dist_sq = 2 * (torch.abs(inner) - self.beta)
        
        sim = -dist_sq
        
        return sim

    def adaptive_weights(self, z_e, z_h):
        
        norm_e = torch.norm(z_e, dim=1, keepdim=True)
        norm_h = torch.norm(z_h, dim=1, keepdim=True)
        
        sum_norm = norm_e + norm_h + self.eps
        w_e = norm_e / sum_norm
        w_h = 1.0 - w_e

        alpha_e = (w_e + w_e.T) / 2.0
        alpha_h = (w_h + w_h.T) / 2.0
            
        return alpha_e, alpha_h

    def forward(self, embeddings):
        """
        embeddings: (2B, D)
        """
        N = embeddings.shape[0]
        B = N // 2

        # Split
        z_e = embeddings[:, :self.dim_euclid]
        z_h = embeddings[:, self.dim_euclid:]

        # Compute Similarities
        z_e_norm = F.normalize(z_e, dim=-1)
        sim_e = torch.matmul(z_e_norm, z_e_norm.T)
        
        sim_h = self.lorentz_similarity(z_h)

        # Compute Weights
        alpha_e, alpha_h = self.adaptive_weights(z_e, z_h)

        # Fusion
        sim = alpha_e * sim_e + alpha_h * sim_h

        # 5. InfoNCE
        mask = torch.eye(N, device=sim.device).bool()
        sim.masked_fill_(mask, -1e9)

        logits = sim / self.temperature

        labels = torch.arange(B, device=sim.device)
        loss_1 = F.cross_entropy(logits[:B, B:], labels)
        loss_2 = F.cross_entropy(logits[B:, :B], labels)

        loss_nce =  (loss_1 + loss_2) * 0.5

        std = torch.sqrt(embeddings.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1.0 - std))
        
        loss = loss_nce + 0.01 * var_loss

        return loss
    
    