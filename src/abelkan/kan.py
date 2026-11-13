import torch
import torch.nn.functional as F
from torch import nn

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
        self, feature_names: list[str] | None = None, precision: int = 3
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
    """

    def __init__(
        self, d_model: int, out_channels: int, num_modes: int = 3
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.num_modes = int(num_modes)
        self.ml_kernel = MittagLefflerPade34()
        out_dim = self.num_modes + self.out_channels * (self.num_modes + 1)
        self.head = nn.Linear(d_model, out_dim)

    def forward(
        self, h: torch.Tensor, alpha: torch.Tensor, t_eval: torch.Tensor
    ) -> torch.Tensor:
        B = h.shape[0]
        dtype = h.dtype
        raw = self.head(h)
        lambdas = 0.005 + 0.15 * torch.sigmoid(raw[:, : self.num_modes])
        rest = raw[:, self.num_modes :].reshape(B, self.out_channels, self.num_modes + 1)
        y_inf = rest[:, :, 0]
        amps = rest[:, :, 1:]

        if t_eval.ndim == 1:
            t_b = t_eval.to(dtype).unsqueeze(0).expand(B, t_eval.shape[0])
        else:
            t_b = t_eval.to(dtype)
        t_pos = torch.clamp(t_b, min=0.0)
        a_b = alpha.to(dtype).unsqueeze(-1)
        t_pow = torch.where(t_pos > 0.0, t_pos ** a_b, torch.zeros_like(t_pos))

        basis = []
        for k in range(self.num_modes):
            z_k = lambdas[:, k : k + 1] * t_pow
            basis.append(self.ml_kernel(z_k, a_b.squeeze(-1)))
        phi = torch.stack(basis, dim=-1)

        return torch.einsum("btk,bck->btc", phi, amps) + y_inf.unsqueeze(1)


class ComplexMittagLefflerSpectralHead(nn.Module):
    """
    Complex-conjugate Mittag-Leffler spectral decoder supporting both monotonic
    fractional relaxation and damped oscillatory modes with L1 sparsity tracking.
    """

    def __init__(
        self,
        d_model: int,
        out_channels: int,
        num_real_modes: int = 3,
        num_osc_modes: int = 2,
        max_omega: float = 1.2,
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.num_real_modes = int(num_real_modes)
        self.num_osc_modes = int(num_osc_modes)
        self.max_omega = float(max_omega)
        self.ml_kernel = MittagLefflerPade34()
        self._last_osc_amps: torch.Tensor | None = None

        n_params = (
            self.num_real_modes
            + 2 * self.num_osc_modes
            + self.out_channels * (1 + self.num_real_modes + 2 * self.num_osc_modes)
        )
        self.head = nn.Linear(d_model, n_params)

    def oscillation_l1_penalty(self) -> torch.Tensor:
        if self._last_osc_amps is None:
            return torch.tensor(0.0, dtype=torch.float64)
        return torch.abs(self._last_osc_amps).mean()

    def forward(
        self, h: torch.Tensor, alpha: torch.Tensor, t_eval: torch.Tensor
    ) -> torch.Tensor:
        B = h.shape[0]
        dtype = h.dtype
        raw = self.head(h)
        kr = self.num_real_modes
        ko = self.num_osc_modes

        idx = 0
        lambdas = 0.005 + 0.15 * torch.sigmoid(raw[:, idx : idx + kr])
        idx += kr
        gammas = 0.002 + 0.12 * torch.sigmoid(raw[:, idx : idx + ko])
        idx += ko
        omegas = 0.02 + (self.max_omega - 0.02) * torch.sigmoid(raw[:, idx : idx + ko])
        idx += ko

        n_basis = kr + 2 * ko
        rest = raw[:, idx:].reshape(B, self.out_channels, 1 + n_basis)
        y_inf = rest[:, :, 0]
        amps = rest[:, :, 1:]
        self._last_osc_amps = amps[:, :, kr:] if ko > 0 else None

        if t_eval.ndim == 1:
            t_b = t_eval.to(dtype).unsqueeze(0).expand(B, t_eval.shape[0])
        else:
            t_b = t_eval.to(dtype)
        t_pos = torch.clamp(t_b, min=0.0)
        a_b = alpha.to(dtype)
        t_pow = torch.where(
            t_pos > 0.0, t_pos ** a_b.unsqueeze(-1), torch.zeros_like(t_pos)
        )

        basis = []
        for k in range(kr):
            z_k = lambdas[:, k : k + 1] * t_pow
            basis.append(self.ml_kernel(z_k, a_b))
        for m in range(ko):
            z_m = gammas[:, m : m + 1] * t_pow
            env = self.ml_kernel(z_m, a_b)
            ang = omegas[:, m : m + 1] * t_pos
            basis.append(env * torch.cos(ang))
            basis.append(env * torch.sin(ang))

        phi = torch.stack(basis, dim=-1)
        return torch.einsum("btk,bck->btc", phi, amps) + y_inf.unsqueeze(1)


class AdaptiveMoEMittagLefflerHead(nn.Module):
    """
    MoE-routed spectral head combining a strictly monotonic real-pole expert
    and a complex-conjugate oscillatory expert via temperature-scaled gating.
    """

    def __init__(
        self,
        d_model: int,
        out_channels: int,
        num_real_modes: int = 3,
        num_osc_modes: int = 2,
        max_omega: float = 1.2,
        temperature: float = 0.8,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.real_expert = MittagLefflerSpectralHead(
            d_model=d_model, out_channels=out_channels, num_modes=num_real_modes
        )
        self.complex_expert = ComplexMittagLefflerSpectralHead(
            d_model=d_model,
            out_channels=out_channels,
            num_real_modes=num_real_modes,
            num_osc_modes=num_osc_modes,
            max_omega=max_omega,
        )
        self.router = nn.Linear(d_model, 2)
        self._last_gate: torch.Tensor | None = None

    def regularization_loss(self) -> torch.Tensor:
        osc_pen = self.complex_expert.oscillation_l1_penalty()
        if self._last_gate is None:
            return osc_pen
        return osc_pen * self._last_gate[:, 1].mean()

    def forward(
        self, h: torch.Tensor, alpha: torch.Tensor, t_eval: torch.Tensor
    ) -> torch.Tensor:
        gate = torch.softmax(self.router(h) / self.temperature, dim=-1)
        self._last_gate = gate
        y_real = self.real_expert(h, alpha, t_eval)
        y_comp = self.complex_expert(h, alpha, t_eval)
        return (
            gate[:, 0].reshape(-1, 1, 1) * y_real
            + gate[:, 1].reshape(-1, 1, 1) * y_comp
        )


class AnchorResidualMittagLefflerHead(nn.Module):
    """
    Universal Anchor-Residual Mittag-Leffler spectral decoder (abelkan v0.4.0):
        y(t) = (x0 + dy0)
               + g_drift * d_c * (t / T_scale)^beta
               + sum_k c_k * (1 - E_{alpha,1}(-lam_k * t^alpha))
               + g_osc * sum_m E_{alpha,1}(-gam_m * t^alpha) *
                 [a_m * (cos(w_m * t) - 1) + b_m * sin(w_m * t)]
    """

    def __init__(
        self,
        d_model: int,
        out_channels: int,
        num_real_modes: int = 3,
        num_osc_modes: int = 2,
        t_scale: float = 48.0,
        max_omega: float = 1.2,
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.kr = int(num_real_modes)
        self.ko = int(num_osc_modes)
        self.t_scale = float(t_scale)
        self.max_omega = float(max_omega)
        self.ml_kernel = MittagLefflerPade34()
        self._reg = torch.tensor(0.0, dtype=torch.float64)

        n_glob = self.kr + 2 * self.ko + 3
        n_ch = self.out_channels * (2 + self.kr + 2 * self.ko)
        self.head = nn.Linear(d_model, n_glob + n_ch)

    def regularization_loss(self) -> torch.Tensor:
        return self._reg

    def forward(
        self,
        h: torch.Tensor,
        alpha: torch.Tensor,
        t_eval: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        B = h.shape[0]
        dtype = h.dtype
        raw = self.head(h)
        idx = 0
        lambdas = 0.005 + 0.15 * torch.sigmoid(raw[:, idx : idx + self.kr])
        idx += self.kr
        gammas = 0.002 + 0.12 * torch.sigmoid(raw[:, idx : idx + self.ko])
        idx += self.ko
        omegas = 0.02 + (self.max_omega - 0.02) * torch.sigmoid(
            raw[:, idx : idx + self.ko]
        )
        idx += self.ko
        beta_drift = 0.8 + 0.9 * torch.sigmoid(raw[:, idx : idx + 1])
        idx += 1
        g_osc = torch.sigmoid(raw[:, idx : idx + 1] - 0.5)
        idx += 1
        g_drift = torch.sigmoid(raw[:, idx : idx + 1] - 1.0)
        idx += 1

        n_per_ch = 2 + self.kr + 2 * self.ko
        rest = raw[:, idx:].reshape(B, self.out_channels, n_per_ch)
        y0 = x0.to(dtype) + 0.25 * torch.tanh(rest[:, :, 0])
        d_amp = rest[:, :, 1] * g_drift
        r_amps = rest[:, :, 2 : 2 + self.kr]
        o_amps = rest[:, :, 2 + self.kr :] * g_osc.unsqueeze(-1)

        osc_pen = torch.abs(o_amps).mean() if self.ko > 0 else 0.0
        self._reg = 8e-4 * (osc_pen + 0.5 * torch.abs(d_amp).mean())

        if t_eval.ndim == 1:
            t_b = t_eval.to(dtype).unsqueeze(0).expand(B, t_eval.shape[0])
        else:
            t_b = t_eval.to(dtype)
        t_pos = torch.clamp(t_b, min=0.0)
        a_b = alpha.to(dtype)
        t_pow = torch.where(
            t_pos > 0.0, t_pos ** a_b.unsqueeze(-1), torch.zeros_like(t_pos)
        )
        tau_ev = t_pos / self.t_scale
        drift_basis = torch.where(
            tau_ev > 0.0, tau_ev ** beta_drift, torch.zeros_like(tau_ev)
        )

        r_basis = torch.stack(
            [
                1.0 - self.ml_kernel(lambdas[:, k : k + 1] * t_pow, a_b)
                for k in range(self.kr)
            ],
            dim=-1,
        )
        o_list = []
        for m in range(self.ko):
            env = self.ml_kernel(gammas[:, m : m + 1] * t_pow, a_b)
            ang = omegas[:, m : m + 1] * t_pos
            o_list.append(env * (torch.cos(ang) - 1.0))
            o_list.append(env * torch.sin(ang))
        y_out = (
            y0.unsqueeze(1)
            + torch.einsum("bt,bc->btc", drift_basis, d_amp)
            + torch.einsum("btk,bck->btc", r_basis, r_amps)
        )
        if self.ko > 0:
            o_basis = torch.stack(o_list, dim=-1)
            y_out = y_out + torch.einsum("btk,bck->btc", o_basis, o_amps)
        return y_out
