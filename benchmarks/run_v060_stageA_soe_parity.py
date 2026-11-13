import json
import os
import sys

sys.path.insert(0, os.path.abspath("intermediate/pylib"))
sys.path.insert(0, os.path.abspath("intermediate"))
sys.path.insert(0, os.path.abspath("output"))
sys.path.insert(0, os.path.abspath("output/abelkan/src"))

import numpy as np
from scipy.optimize import nnls
import torch
import torch.nn as nn
import torch.nn.functional as F

from abelkan import AnchorResidualMittagLefflerHead
from abelkan.attention import AbelCausalAttention
import fast_data_helper as base

OUT_DIR = "output"


# ---------------------------------------------------------------------------
# 1. Kernel Approximation Parity: (1 + dt)^(-c) via Sum of M Exponentials
# ---------------------------------------------------------------------------
def fit_soe_power_law(c: float, M: int, dt_max: float = 4096.0):
    dt_grid = np.geomspace(1.0, dt_max, 512)
    lambdas = np.geomspace(0.5 / dt_max, 2.0, M)
    A = np.exp(-dt_grid[:, None] * lambdas[None, :])
    y = (1.0 + dt_grid) ** (-c)
    weights = 1.0 / np.maximum(y, 1e-4)
    w, _ = nnls(A * weights[:, None], y * weights)
    y_hat = A @ w
    rel_err = np.abs(y_hat - y) / y
    l2_rel = float(np.linalg.norm(y_hat - y) / np.linalg.norm(y))
    return {
        "c": c,
        "M": M,
        "l2_rel_err": round(l2_rel, 6),
        "max_rel_err": round(float(np.max(rel_err)), 6),
        "mean_rel_err": round(float(np.mean(rel_err)), 6),
    }


# ---------------------------------------------------------------------------
# 2. SOE-Abel Causal Attention Module (O(L * M) Recurrent Scan)
# ---------------------------------------------------------------------------
class SOEAbelCausalAttention(nn.Module):
    def __init__(self, d_model: int, M: int = 8, time_scale: float = 150.0, alpha_min: float = 0.40, alpha_max: float = 0.92):
        super().__init__()
        self.d_model = int(d_model)
        self.M = int(M)
        self.time_scale = float(time_scale)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.alpha_head = nn.Linear(d_model, 2)
        self.raw_gamma = nn.Parameter(torch.tensor([0.85]))

        lams = np.geomspace(1.0 / (2.0 * time_scale), 2.0, M).astype(np.float64)
        self.lambdas = torch.tensor(lams, dtype=torch.float64)
        self.mode_logits = nn.Parameter(torch.zeros(M, dtype=torch.float64))

    def forward(self, x, timestamps, attn_mask=None, alpha=None):
        B, N, D = x.shape
        dtype = x.dtype
        device = x.device
        valid = torch.ones((B, N), dtype=torch.bool, device=device) if attn_mask is None else (attn_mask > 0.5)

        idx = torch.arange(N, device=device).unsqueeze(0)
        causal_mask = (idx.unsqueeze(2) >= idx.unsqueeze(1)) & valid.unsqueeze(1)
        causal_w = causal_mask.to(dtype)
        prefix_mean = torch.matmul(causal_w, x) / torch.clamp(causal_w.sum(dim=-1, keepdim=True), min=1.0)
        alpha_raw = self.alpha_head(prefix_mean)
        sigma_alpha = 0.01 + 0.15 * F.softplus(alpha_raw[..., 1])
        if alpha is not None:
            alpha_hat = alpha.to(dtype).unsqueeze(-1).expand(B, N) if alpha.ndim == 1 else alpha.to(dtype)
        else:
            alpha_hat = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(alpha_raw[..., 0])

        Q = F.softplus(self.q_proj(x)) + 1e-3
        K = F.softplus(self.k_proj(x)) + 1e-3
        V = self.v_proj(x)

        lams = self.lambdas.to(dtype)
        slow_bias = torch.linspace(1.0, 0.0, self.M, dtype=dtype)
        gamma = F.softplus(self.raw_gamma)

        S = torch.zeros((B, self.M, D, D), dtype=dtype, device=device)
        Z = torch.zeros((B, self.M, D), dtype=dtype, device=device)
        outs = []
        for t in range(N):
            dt_step = torch.zeros((B, 1), dtype=dtype, device=device) if t == 0 else torch.clamp(timestamps[:, t] - timestamps[:, t - 1], min=0.0).unsqueeze(-1)
            decay = torch.exp(-dt_step * lams.unsqueeze(0))  # (B, M)
            v_mask = valid[:, t].to(dtype).unsqueeze(-1)  # (B, 1)

            kt = K[:, t, :]
            vt = V[:, t, :]
            qt = Q[:, t, :]

            kv = kt.unsqueeze(-1) * vt.unsqueeze(-2)
            S = S * decay.unsqueeze(-1).unsqueeze(-1) + (v_mask.unsqueeze(-1).unsqueeze(-1) * kv.unsqueeze(1))
            Z = Z * decay.unsqueeze(-1) + (v_mask.unsqueeze(-1) * kt.unsqueeze(1))

            num_m = torch.matmul(qt.unsqueeze(1).unsqueeze(2), S).squeeze(2)
            den_m = torch.clamp(torch.sum(qt.unsqueeze(1) * Z, dim=-1), min=1e-6).unsqueeze(-1)
            read_m = num_m / den_m

            a_t = (1.0 - alpha_hat[:, t]).unsqueeze(-1)
            w_t = torch.softmax(self.mode_logits.unsqueeze(0) + gamma * a_t * slow_bias.unsqueeze(0), dim=-1)
            out_t = torch.sum(w_t.unsqueeze(-1) * read_m, dim=1)
            outs.append(out_t)

        out = self.out_proj(torch.stack(outs, dim=1))
        return out, alpha_hat, sigma_alpha


class ArchAbelParityModel(nn.Module):
    def __init__(self, c_in, c_out, d_model=36, mode="dense", M=8, time_scale=48.0):
        super().__init__()
        self.emb = nn.Linear(c_in * 2 + 1, d_model)
        if mode == "dense":
            self.attn = AbelCausalAttention(d_model, decay_mode="abel", time_scale=time_scale)
        else:
            self.attn = SOEAbelCausalAttention(d_model, M=M, time_scale=time_scale)
        self.norm = nn.LayerNorm(d_model)
        self.head = AnchorResidualMittagLefflerHead(d_model, c_out, num_real_modes=3, num_osc_modes=2, t_scale=time_scale)
        self.cls_head = nn.Linear(d_model, 1)

    def forward(self, vals, mask, tobs, t_eval):
        B, L, _ = vals.shape
        x_in = torch.cat([vals, mask, (tobs / max(float(t_eval[-1].item()), 1.0)).unsqueeze(-1)], dim=-1)
        h = torch.tanh(self.emb(x_in))
        att, a_hat, _ = self.attn(h, tobs, attn_mask=mask.mean(dim=-1))
        h = self.norm(h + att)
        h_pool = h[:, -1, :]
        a_pool = a_hat.mean(dim=-1)
        x0 = vals[:, 0, :]
        pred = self.head(h_pool, a_pool, t_eval, x0)
        reg = torch.tensor(0.0, dtype=pred.dtype)
        logit = self.cls_head(h_pool).squeeze(-1)
        return pred, logit


def main():
    print("=== Part 2 Stage A.1: Kernel Approximation Parity ===")
    kernel_results = []
    for c in [0.3, 0.5, 0.7]:
        for M in [4, 6, 8, 12, 16]:
            res = fit_soe_power_law(c=c, M=M, dt_max=4096.0)
            kernel_results.append(res)
            print(f"  c={c:.1f}, M={M:2d} | L2_rel={res['l2_rel_err']:.6f} | Mean_rel={res['mean_rel_err']:.6f} | Max_rel={res['max_rel_err']:.6f}")

    # Build Burst-Gap 65% ICU identically to v0.5.0
    rng_ext = np.random.default_rng(2026)
    N_bg = base.N_tot
    vals_bg = np.zeros((N_bg, 5, 3))
    mask_bg = np.ones((N_bg, 5, 3))
    tobs_bg = np.zeros((N_bg, 5))
    idx_bg = np.array([0, 1, 2, 3, 12])
    for i in range(N_bg):
        tobs_bg[i] = base.t_a[idx_bg]
        vals_bg[i] = base.y_a[i, idx_bg, :] * (1.0 + rng_ext.normal(0.0, 0.08, size=(5, 3)))

    TASKS = [
        ("PhysioNet ICU-48h", base.vals_a, base.mask_a, base.tobs_a, base.y_a, base.risk_a, base.t_a, 13, 48.0, 210, True),
        ("Burst-Gap 65% ICU", vals_bg, mask_bg, tobs_bg, base.y_a, base.risk_a, base.t_a, 13, 48.0, 210, True),
        ("Circadian 96h", base.vals_c, base.mask_c, base.tobs_c, base.y_c, np.zeros(base.N_tot), base.t_c, 21, 96.0, 210, False),
    ]
    VARIANTS = [
        ("Dense O(L^2) Abel", "dense", 0),
        ("SOE-Abel (M=6)", "soe", 6),
        ("SOE-Abel (M=8)", "soe", 8),
        ("SOE-Abel (M=12)", "soe", 12),
    ]
    SEEDS = [42, 101, 777]
    task_parity = {v[0]: {} for v in VARIANTS}

    print("\n=== Part 2 Stage A.2: End-to-End 3-Seed Task Parity ===")
    for tname, vals, mask, tobs, y, cy, tev, oend, ts, ntr, isc in TASKS:
        print(f"\n--- Task: {tname} ---")
        ic, oc = vals.shape[-1], y.shape[-1]
        for vname, mode, M in VARIANTS:
            r2s, lats = [], []
            for sd in SEEDS:
                torch.manual_seed(sd)
                base.rng = np.random.default_rng(sd)
                out = base.train_and_eval_arch(
                    lambda m=mode, mm=M: ArchAbelParityModel(ic, oc, d_model=36, mode=m, M=mm, time_scale=ts),
                    vals[:ntr], mask[:ntr], tobs[:ntr], y[:ntr], cy[:ntr],
                    vals[ntr:], mask[ntr:], tobs[ntr:], y[ntr:], cy[ntr:],
                    tev, oend, epochs=55, lr=4.5e-3, is_cls=isc,
                )
                r2s.append(out["r2_extrap"])
                lats.append(out["ms_per_sample"])
            mr2, sr2 = float(np.mean(r2s)), float(np.std(r2s))
            mlat = float(np.mean(lats))
            task_parity[vname][tname] = {"mean_r2": round(mr2, 4), "std_r2": round(sr2, 4), "lat_ms": round(mlat, 3)}
            print(f"  {vname:20s} | R2 = {mr2:+.4f} +/- {sr2:.3f} | Lat = {mlat:.3f} ms")

    with open(os.path.join(OUT_DIR, "v060_stageA_soe_parity.json"), "w", encoding="utf-8") as f:
        json.dump({"kernel_parity": kernel_results, "task_parity": task_parity}, f, indent=2)
    print("\nSaved output/v060_stageA_soe_parity.json!")


if __name__ == "__main__":
    main()
