from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pade import MittagLefflerPade34


class GaussianBellGate(nn.Module):
    """Optional Gaussian tolerance bell exp(-beta * (x - center)^2)."""

    def __init__(self, init_center: float = 0.5, init_beta: float = 2.5) -> None:
        super().__init__()
        self.center = nn.Parameter(torch.tensor([float(init_center)]))
        self.raw_beta = nn.Parameter(torch.tensor([float(init_beta)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = F.softplus(self.raw_beta)
        return torch.exp(-beta * ((x - self.center) ** 2))


class LogExpMultKAN(nn.Module):
    """
    Generalized multiplicative log-exponential KAN layer for power-law discovery:
        y = sum_{b=1}^{B} w_b * exp(sum_{c=1}^{D} W_{b, c} * ln(clamp(x_c, min=eps)))
    """

    def __init__(
        self, in_features: int, num_branches: int = 2, eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.num_branches = int(num_branches)
        self.eps = float(eps)
        self.W_power = nn.Parameter(torch.zeros(num_branches, in_features))
        self.raw_branch_weights = nn.Parameter(torch.zeros(num_branches))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_x = torch.log(torch.clamp(x, min=self.eps))
        exponents = torch.matmul(log_x, self.W_power.transpose(0, 1))
        branch_vals = torch.exp(torch.clamp(exponents, min=-15.0, max=15.0))
        w = torch.softmax(self.raw_branch_weights, dim=0)
        return (branch_vals * w).sum(dim=-1)

    def l1_loss(self) -> torch.Tensor:
        return torch.abs(self.W_power).sum()

    def to_symbolic(
        self, feature_names: Optional[List[str]] = None, precision: int = 3
    ) -> str:
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.in_features)]
        w = torch.softmax(self.raw_branch_weights, dim=0)
        terms = []
        for b in range(self.num_branches):
            wb = float(w[b].item())
            if wb < 1e-3:
                continue
            factors = []
            for c, fname in enumerate(feature_names):
                p = float(self.W_power[b, c].item())
                if abs(p) >= 10 ** (-precision):
                    factors.append(f"{fname}^({p:.{precision}f})")
            expr = " * ".join(factors) if factors else "1"
            terms.append(f"{wb:.{precision}f} * ({expr})")
        return " + ".join(terms) if terms else "0"


class MittagLefflerSpectralHead(nn.Module):
    """
    Universal Mittag-Leffler spectral trajectory decoder:
        y_c(t) = y_inf_c + sum_{k=1}^{K} c_{c, k} * E_{alpha, 1}(-lam_k * t^alpha)
    for arbitrary `out_channels`.
    """

    def __init__(
        self, d_model: int, out_channels: int, num_modes: int = 3
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.num_modes = int(num_modes)
        self.ml_kernel = MittagLefflerPade34()
        # Predict num_modes decay rates + out_channels*(num_modes + 1) amplitudes & asymptotes
        out_dim = self.num_modes + self.out_channels * (self.num_modes + 1)
        self.head = nn.Linear(d_model, out_dim)

    def forward(
        self, h: torch.Tensor, alpha: torch.Tensor, t_eval: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h: (B, d_model)
            alpha: (B,)
            t_eval: (T,) or (B, T)
        Returns:
            trajectories: (B, T, out_channels)
        """
        B = h.shape[0]
        raw = self.head(h)
        lambdas = 0.005 + 0.15 * torch.sigmoid(raw[:, : self.num_modes])  # (B, K)
        rest = raw[:, self.num_modes :].reshape(B, self.out_channels, self.num_modes + 1)
        y_inf = rest[:, :, 0]  # (B, C)
        amps = rest[:, :, 1:]  # (B, C, K)

        if t_eval.ndim == 1:
            t_b = t_eval.unsqueeze(0).expand(B, t_eval.shape[0])
        else:
            t_b = t_eval
        t_pos = torch.clamp(t_b, min=0.0)
        a_b = alpha.unsqueeze(-1)
        t_pow = torch.where(t_pos > 0.0, t_pos ** a_b, torch.zeros_like(t_pos))  # (B, T)

        basis = []
        for k in range(self.num_modes):
            z_k = lambdas[:, k : k + 1] * t_pow
            basis.append(self.ml_kernel(z_k, alpha))
        phi = torch.stack(basis, dim=-1)  # (B, T, K)

        traj = torch.einsum("btk,bck->btc", phi, amps) + y_inf.unsqueeze(1)
        return traj
