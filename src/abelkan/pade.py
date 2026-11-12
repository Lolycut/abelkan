import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class MittagLefflerPade34(nn.Module):
    """
    Differentiable global [3/4] Pade approximant of the one-parameter
    Mittag-Leffler function E_{alpha, 1}(-z) on z >= 0, alpha in (0.25, 1.0].

    Matches Maclaurin expansion as z -> 0 and Poincare asymptotics as z -> inf
    via Euler's reflection identity 1/Gamma(1 - k*alpha) = Gamma(k*alpha)*sin(k*pi*alpha)/pi,
    eliminating the Gamma pole at alpha = 0.50. The 5x5 linear system is solved in
    float64 and cast back to the input dtype.
    """

    def __init__(self, bridge_start: float = 0.90) -> None:
        super().__init__()
        self.bridge_start = float(bridge_start)

    @staticmethod
    def _solve_pade_coeffs(
        alpha: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        a = alpha.to(torch.float64)
        pi = math.pi

        g1 = torch.exp(-torch.lgamma(1.0 + a))
        g2 = torch.exp(-torch.lgamma(1.0 + 2.0 * a))
        g3 = torch.exp(-torch.lgamma(1.0 + 3.0 * a))
        g4 = torch.exp(-torch.lgamma(1.0 + 4.0 * a))
        g5 = torch.exp(-torch.lgamma(1.0 + 5.0 * a))

        m1 = torch.exp(torch.lgamma(a)) * torch.sin(pi * a) / pi
        m2 = torch.exp(torch.lgamma(2.0 * a)) * torch.sin(2.0 * pi * a) / pi

        zeros = torch.zeros_like(a)
        ones = torch.ones_like(a)

        row0 = torch.stack([g2, -g1, ones, zeros, -ones], dim=-1)
        row1 = torch.stack([-g3, g2, -g1, ones, zeros], dim=-1)
        row2 = torch.stack([g4, -g3, g2, -g1, zeros], dim=-1)
        row3 = torch.stack([zeros, zeros, zeros, -m1, ones], dim=-1)
        row4 = torch.stack([-g1, ones, -m1, m2, zeros], dim=-1)

        A = torch.stack([row0, row1, row2, row3, row4], dim=-2)
        b = torch.stack([g3, -g4, g5, zeros, -g2], dim=-1)

        sol = torch.linalg.solve(A, b)
        q1, q2, q3, q4, p3 = sol[..., 0], sol[..., 1], sol[..., 2], sol[..., 3], sol[..., 4]
        p1 = q1 - g1
        p2 = q2 - g1 * q1 + g2
        return p1, p2, p3, q1, q2, q3, q4

    def _eval_pade_raw(self, z64: torch.Tensor, alpha64: torch.Tensor) -> torch.Tensor:
        p1, p2, p3, q1, q2, q3, q4 = self._solve_pade_coeffs(alpha64)
        z2 = z64 * z64
        z3 = z2 * z64
        z4 = z2 * z2
        num = 1.0 + p1 * z64 + p2 * z2 + p3 * z3
        den = 1.0 + q1 * z64 + q2 * z2 + q3 * z3 + q4 * z4
        return num / torch.clamp(den, min=1e-12)

    def forward(self, z: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        orig_dtype = z.dtype
        z64 = torch.clamp(z.to(torch.float64), min=0.0)
        a64 = torch.clamp(alpha.to(torch.float64), min=0.25, max=1.0)
        while a64.ndim < z64.ndim:
            a64 = a64.unsqueeze(-1)

        bs = self.bridge_start
        a_clamped = torch.clamp(a64, max=bs)
        val_pade = self._eval_pade_raw(z64, a_clamped)
        val_exp = torch.exp(-z64)

        s = torch.clamp((a64 - bs) / (1.0 - bs), min=0.0, max=1.0)
        w = s * s * (3.0 - 2.0 * s)
        out64 = (1.0 - w) * val_pade + w * val_exp
        return out64.to(orig_dtype)

    def caputo_basis_and_derivative(
        self, lam: torch.Tensor, t: torch.Tensor, alpha: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates y(t) = E_{alpha,1}(-lam * t^alpha) and its exact Caputo derivative
        D_t^alpha y(t) = -lam * E_{alpha,1}(-lam * t^alpha).
        """
        t_pos = torch.clamp(t, min=0.0)
        a_exp = alpha
        while a_exp.ndim < t_pos.ndim:
            a_exp = a_exp.unsqueeze(-1)
        lam_exp = lam
        while lam_exp.ndim < t_pos.ndim:
            lam_exp = lam_exp.unsqueeze(-1)
        z = lam_exp * torch.where(t_pos > 0.0, t_pos ** a_exp, torch.zeros_like(t_pos))
        e_val = self.forward(z, alpha)
        return e_val, -lam_exp * e_val

    def certify_pole_free(
        self, alpha_grid: Optional[torch.Tensor] = None
    ) -> Dict[str, object]:
        """
        Algebraically certifies pole-freedom of Q_4(z) on [0, +inf) via Descartes' Rule
        of Signs: if q1, q2, q3, q4 > 0 for all alpha in alpha_grid, Q_4(z) has zero
        positive real roots on [0, +inf).
        """
        if alpha_grid is None:
            alpha_grid = torch.linspace(0.40, self.bridge_start, 501, dtype=torch.float64)
        _, _, _, q1, q2, q3, q4 = self._solve_pade_coeffs(alpha_grid.to(torch.float64))
        min_q1 = float(q1.min().item())
        min_q2 = float(q2.min().item())
        min_q3 = float(q3.min().item())
        min_q4 = float(q4.min().item())
        certified = (min_q1 > 0.0) and (min_q2 > 0.0) and (min_q3 > 0.0) and (min_q4 > 0.0)
        return {
            "certified_pole_free": certified,
            "min_q1": min_q1,
            "min_q2": min_q2,
            "min_q3": min_q3,
            "min_q4": min_q4,
            "n_alphas": int(alpha_grid.numel()),
        }


def mittag_leffler_pade34(z: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Functional wrapper for MittagLefflerPade34."""
    return MittagLefflerPade34()(z, alpha)
