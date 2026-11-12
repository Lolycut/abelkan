import torch
from torch import nn


class TemperatureTopKMoE(nn.Module):
    """
    Sparse Top-k Mixture-of-Experts with temperature-annealed softmax gating
    and load-balancing regularization.
    """

    def __init__(
        self,
        d_model: int,
        experts: list[nn.Module] | None = None,
        num_experts: int = 4,
        k: int = 2,
        init_temp: float = 2.0,
        gate_in_dim: int | None = None,
    ) -> None:
        super().__init__()
        if experts is not None:
            self.experts = nn.ModuleList(experts)
            self.num_experts = len(experts)
        else:
            self.experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_experts)])
            self.num_experts = int(num_experts)

        self.k = min(int(k), self.num_experts)
        self.router_temp = float(init_temp)
        in_dim = int(gate_in_dim) if gate_in_dim is not None else int(d_model)
        self.router = nn.Linear(in_dim, self.num_experts)
        self._last_balance_loss = torch.tensor(0.0)

    def set_temperature(self, tau: float) -> None:
        self.router_temp = max(0.1, float(tau))

    def balance_loss(self) -> torch.Tensor:
        return self._last_balance_loss

    def forward(
        self, x: torch.Tensor, gate_features: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, d_model)
            gate_features: optional (B, gate_in_dim); defaults to x
        Returns:
            out: (B, d_model)
            top_k_indices: (B, k)
        """
        g_in = gate_features if gate_features is not None else x
        logits = self.router(g_in) / max(0.1, self.router_temp)
        probs = torch.softmax(logits, dim=-1)

        top_vals, top_idx = torch.topk(probs, k=self.k, dim=-1)
        top_norm = top_vals / torch.clamp(top_vals.sum(dim=-1, keepdim=True), min=1e-8)

        # Load-balancing loss: N * sum_e f_e * P_e
        eye = torch.eye(self.num_experts, dtype=probs.dtype, device=probs.device)
        one_hot = eye[top_idx].sum(dim=1)
        f_e = one_hot.mean(dim=0)
        p_e = probs.mean(dim=0)
        self._last_balance_loss = self.num_experts * (f_e * p_e).sum()

        out = x.clone()
        for slot in range(self.k):
            idx_s = top_idx[:, slot]
            w_s = top_norm[:, slot : slot + 1]
            expert_out = torch.zeros_like(x)
            for e_id, exp_mod in enumerate(self.experts):
                sel = idx_s == e_id
                if sel.any():
                    expert_out[sel] = torch.nn.functional.gelu(exp_mod(x[sel]))
            out = out + w_s * expert_out

        return out, top_idx
