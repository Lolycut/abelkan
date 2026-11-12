import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbelCausalAttention(nn.Module):
    """
    Drop-in causal self-attention for irregular time series with selectable temporal bias:
      - "abel": heavy-tailed power-law decay -softplus(gamma) * (1 - alpha_i) * ln(eps + dt)
      - "exp_alibi": Markovian exponential decay -softplus(gamma) * dt / time_scale
      - "none": standard causal dot-product attention without temporal decay

    If `alpha` is None, a lightweight pre-attention head predicts causal `alpha_hat_i`
    and epistemic uncertainty `sigma_alpha` from the cumulative causal prefix mean of `x`.
    """

    def __init__(
        self,
        d_model: int,
        decay_mode: Literal["abel", "exp_alibi", "none"] = "abel",
        eps: float = 1.0,
        time_scale: float = 150.0,
        alpha_min: float = 0.40,
        alpha_max: float = 0.92,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.decay_mode = decay_mode
        self.eps = float(eps)
        self.time_scale = float(time_scale)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.raw_gamma = nn.Parameter(torch.tensor([0.85]))
        self.alpha_head = nn.Linear(d_model, 2)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, d_model)
            timestamps: (B, N) monotonic or irregular timestamps
            attn_mask: optional boolean/binary mask (B, N), 1 for valid tokens
            alpha: optional external fractional order (B,) or (B, N) for weight tying
        Returns:
            out: (B, N, d_model)
            alpha_hat: (B, N)
            sigma_alpha: (B, N)
        """
        B, N, _ = x.shape
        device = x.device
        dtype = x.dtype

        if attn_mask is None:
            valid = torch.ones((B, N), dtype=torch.bool, device=device)
        else:
            valid = attn_mask > 0.5

        idx = torch.arange(N, device=device).unsqueeze(0)
        idx_causal = idx.unsqueeze(2) >= idx.unsqueeze(1)
        time_causal = (timestamps.unsqueeze(2) >= timestamps.unsqueeze(1) - 1e-9) & idx_causal
        causal_mask = time_causal & valid.unsqueeze(1)

        causal_w = causal_mask.to(dtype)
        prefix_mean = torch.matmul(causal_w, x) / torch.clamp(
            causal_w.sum(dim=-1, keepdim=True), min=1.0
        )
        alpha_raw = self.alpha_head(prefix_mean)
        sigma_alpha = 0.01 + 0.15 * F.softplus(alpha_raw[..., 1])

        if alpha is not None:
            a_in = alpha.to(dtype)
            if a_in.ndim == 1:
                alpha_hat = a_in.unsqueeze(-1).expand(B, N)
            else:
                alpha_hat = a_in
        else:
            span = self.alpha_max - self.alpha_min
            alpha_hat = self.alpha_min + span * torch.sigmoid(alpha_raw[..., 0])

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(float(self.d_model))

        if self.decay_mode != "none":
            dt = torch.clamp(timestamps.unsqueeze(2) - timestamps.unsqueeze(1), min=0.0)
            gamma = F.softplus(self.raw_gamma)
            if self.decay_mode == "abel":
                bias = -gamma * (1.0 - alpha_hat.unsqueeze(-1)) * torch.log(self.eps + dt)
            elif self.decay_mode == "exp_alibi":
                bias = -gamma * (dt / self.time_scale)
            else:
                raise ValueError(f"Unknown decay_mode: {self.decay_mode}")
            scores = scores + bias

        scores_masked = torch.where(causal_mask, scores, torch.tensor(-1e9, dtype=dtype, device=device))
        attn_w = torch.softmax(scores_masked, dim=-1) * causal_w
        attn_w = attn_w / torch.clamp(attn_w.sum(dim=-1, keepdim=True), min=1e-12)

        out = self.out_proj(torch.matmul(attn_w, V))
        return out, alpha_hat, sigma_alpha


class AbelTransformerLayer(nn.Module):
    """Residual Transformer block wrapping AbelCausalAttention + LayerNorm."""

    def __init__(
        self,
        d_model: int,
        decay_mode: Literal["abel", "exp_alibi", "none"] = "abel",
        eps: float = 1.0,
        time_scale: float = 150.0,
    ) -> None:
        super().__init__()
        self.attn = AbelCausalAttention(
            d_model=d_model, decay_mode=decay_mode, eps=eps, time_scale=time_scale
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        att_out, alpha_hat, sigma_alpha = self.attn(x, timestamps, attn_mask=attn_mask, alpha=alpha)
        h = self.norm1(x + att_out)
        h = self.norm2(h + self.ffn(h))
        return h, alpha_hat, sigma_alpha
