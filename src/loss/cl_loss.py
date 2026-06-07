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
        gamma_power=0.5, # 论文推荐 0.5 (即 sqrt)
        # temperature=0.1,
        temperature=0.2,
        # temperature=0.3,
        eps=1e-6,        # 稍微调大一点防止除零
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
        
        # [数值稳定] 防止 cosh 溢出，双曲函数超过 15 后数值会急剧膨胀，最终导致浮点数溢出
        # (虽然 gamma 已经压缩了，但防一手总是好的)
        r_tilde = r_tilde.clamp(max=15.0) 

        direction = z / r

        # x0^2 - x_rest^2 = beta
        sqrt_beta = self.beta ** 0.5
        x0 = sqrt_beta * torch.cosh(r_tilde)
        x_rest = sqrt_beta * torch.sinh(r_tilde) * direction

        return torch.cat([x0, x_rest], dim=-1)

    def lorentz_similarity(self, z):
        """
        计算 Lorentz 相似度并归一化到 [-1, 1]
        """
        z_l = self.to_lorentz(z) # (2B, D+1)
        
        # 1. 计算 Lorentz 内积 <x, y>
        # (2B, 1, D+1) * (1, 2B, D+1) -> (2B, 2B) 广播计算
        x = z_l.unsqueeze(1)
        y = z_l.unsqueeze(0)
        
        # <x, y> = -x0*y0 + x_rest*y_rest
        inner = -x[..., 0] * y[..., 0] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
        
        # 2. 转换为距离 (Lorentz Distance squared approx)
        # 内积一定是负数 (<= -beta)
        # 距离 d^2 ≈ 2 * (|inner| - beta)
        dist_sq = 2 * (torch.abs(inner) - self.beta)
        
        # 3. 转换为相似度
        # 使用 -dist_sq 是最优雅的方案
        # sim = -dist_sq
        sim = 1 - torch.tanh(dist_sq)  # 归一化到 [-1, 1]
        
        return sim

    def adaptive_weights(self, z_e, z_h):
        """
        基于模长的自适应权重 (无需参数)
        """
        # 原始模长越大 -> 说明在那个维度的特征越显著 -> 权重越大
        
        
        norm_e = torch.norm(z_e, dim=1, keepdim=True)
        norm_h = torch.norm(z_h, dim=1, keepdim=True)
        
        # 简单的比例分配
        sum_norm = norm_e + norm_h + self.eps
        w_e = norm_e / sum_norm
        w_h = 1.0 - w_e
        
        # 广播成矩阵权重
        alpha_e = (w_e + w_e.T) / 2.0
        alpha_h = (w_h + w_h.T) / 2.0
            
        return alpha_e, alpha_h

    def forward(self, embeddings):
        """
        embeddings: (2B, D)
        """
        N = embeddings.shape[0]
        B = N // 2

        # 1. Split
        z_e = embeddings[:, :self.dim_euclid]
        z_h = embeddings[:, self.dim_euclid:]

        # 2. Compute Similarities
        # 欧氏部分: Cosine [-1, 1]
        z_e_norm = F.normalize(z_e, dim=-1)
        sim_e = torch.matmul(z_e_norm, z_e_norm.T)
        
        # 双曲部分: Normalized Lorentz Sim [-1, 1]
        sim_h = self.lorentz_similarity(z_h)

        # 3. Compute Weights
        alpha_e, alpha_h = self.adaptive_weights(z_e, z_h)

        # 4. Fusion
        # 现在 sim_e 和 sim_h 都在 [-1, 1] 范围内，可以直接线性加权
        sim = alpha_e * sim_e + alpha_h * sim_h

        # 5. InfoNCE
        mask = torch.eye(N, device=sim.device).bool()
        sim.masked_fill_(mask, -1e9)

        logits = sim / self.temperature

        labels = torch.arange(B, device=sim.device)
        loss_1 = F.cross_entropy(logits[:B, B:], labels)
        loss_2 = F.cross_entropy(logits[B:, :B], labels)

        loss_nce =  (loss_1 + loss_2) * 0.5

        # ===== variance regularization (防塌缩) =====
        std = torch.sqrt(embeddings.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1.0 - std))

        # 系数先用 0.01
        loss = loss_nce + 0.01 * var_loss

        return loss
    
    