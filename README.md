# `abelkan` v0.6.0 — Fractional Mittag-Leffler KAN & Power-Law $O(L)$ Attention

[![PyPI](https://img.shields.io/badge/PyPI-abelkan%200.6.0-065F46)](https://pypi.org/project/abelkan/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0005--5503--3269-A6CE39)](https://orcid.org/0009-0005-5503-3269)

**`abelkan`** is a PyTorch library for **non-Markovian, irregular-time, and long-context sequence modeling**, extracted from the [TerraKAN](https://github.com/Lolycut/TerraKAN) (`Frac-ML-KAT-MoE`) research framework. It combines:

1. **Routh–Hurwitz Certified `[3/4]` Padé Mittag-Leffler Kernels** (`MittagLefflerPade34`, `certify_complex_stability()`): zero poles on the closed right half-plane $\{\mathrm{Re}(z) \ge 0\}$ across all $\alpha \in [0.40, 0.90]$ ($\max \mathrm{Re}(z) = -1.3230 < 0$).
2. **Anchor-Residual Spectral Decoder (`abelkan-v5-Universal`)**: #1 overall across **14 architectures × 8 tasks × 3 seeds** (`+0.373` Mean $R^2$; `+0.663` on 7 non-battery tasks).
3. **$O(L)$ Sum-of-Exponentials Power-Law Attention (`SOE-Abel`, $M=8$)**: exact $O(\log L)$ Blelloch associative & chunkwise scans (`< 1.4e-15` parity with sequential scan), **492× lower tail error** in `bfloat16` via `bf16-IO + fp32-acc`, **85.94% recall at $8\times$ extrapolation (`L=1024`)** on 15.58M-parameter `MQAR`, and scaling up to **`L = 32,768`** on a 4–6 GB laptop **RTX 3050** (`2.65 GB` peak VRAM).

---

## Version Comparison (`v0.4.0` → `v0.5.0` → `v0.6.0`)

| Version | Key Architecture & Features | Complexity / Precision | Headline Verified Result |
| :--- | :--- | :--- | :--- |
| **`v0.4.0`** | `AbelCausalAttention`, `MittagLefflerPade34`, `AnchorResidualMittagLefflerHead`, `FracKANLayer` | $O(L^2)$ dense attention (`fp32`/`fp64`) | Unifies medical, clinical, and climate extrapolation without per-domain manual switching |
| **`v0.5.0`** | `certify_complex_stability()` (Routh–Hurwitz $Q_4(z)$), `abelkan-v5-AutoMoE`, 14-arch 3-seed suite | Verified Hurwitz stability (`501/501` $\alpha$ grid) + CPU `SOE` prototype (`46.4×` at `L=4096`) | **#1 overall** (`+0.373` Mean $R^2$) vs `ODE-RNN` (`+0.354`), `iTransformer+RevIN` (`+0.193`), `PI-KAN-PointNet`, `DeepOKAN` |
| **`v0.6.0`** *(Current)* | `abelkan.soe` (`blelloch_associative_scan`, `soe_chunkwise_scan`), `abelkan.triton_soe` (`bf16-IO + fp32-acc`), RTX 3050 15.5M `MQAR` & `32K` benchmarks | $O(L)$ chunkwise / $O(\log L)$ Blelloch scan; `bf16` I/O + `fp32` state accumulation | **85.94%** (`8×`) & **41.41%** (`16×`) on 15.58M `MQAR` (RTX 3050); **`2.65 GB` VRAM at `L=32,768`** (zero `OOM`) |

---

## Benchmark 1: 14-Architecture, 3-Seed Multi-Domain Extrapolation (`v0.5.0`)

All models are supervised strictly on $[0, T_{\mathrm{obs}}]$ and evaluated on blind out-of-time extrapolation $(T_{\mathrm{obs}}, T_{\mathrm{end}}]$ across 3 seeds (`42, 101, 777`):

| Architecture | Params | Lat. (ms) | Mean $R^2$ | MRSA | Wt 45d | NOAA 12y | Minsk 4-Ch | ICU-48h | Circad. 96h | Battery | Burst 65% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`abelkan-v5-Universal`** | 7,422 | 0.137 | **+0.373** | +0.964 | +0.652 | +0.459 | +0.319 | +0.683 | **+0.763** | -1.660 | +0.807 |
| `ODE-RNN (2019)` | 2,788 | 0.211 | +0.354 | +0.661 | +0.791 | **+0.664** | +0.082 | -0.017 | +0.040 | +0.293 | +0.321 |
| **`abelkan-v5-ClimTrend`** | 6,912 | 0.439 | +0.352 | +0.924 | +0.503 | +0.134 | +0.406 | +0.603 | +0.343 | -0.664 | +0.566 |
| **`abelkan-v5-AutoMoE`** | 7,459 | 0.184 | +0.241 | +0.934 | +0.744 | -0.949 | +0.352 | +0.644 | +0.752 | -1.264 | +0.714 |
| `iTransformer+RevIN` | 8,836 | 0.161 | +0.193 | +0.842 | +0.779 | -1.632 | +0.035 | +0.586 | +0.388 | +0.001 | +0.545 |
| **`abelkan-v5-Med`** | 7,262 | 0.150 | +0.172 | **+0.988** | +0.195 | +0.133 | -0.253 | **+0.810** | +0.403 | -1.737 | **+0.834** |
| `FNO-1D+Detrend` | 5,500 | 0.138 | +0.133 | +0.803 | +0.653 | +0.553 | -0.638 | +0.033 | -1.493 | **+0.594** | +0.560 |
| `PI-KAN-PointNet (2025)` | 9,202 | 0.467 | -3.499 | +0.969 | **+0.863** | -13.2 | +0.028 | +0.542 | +0.141 | -17.8 | +0.516 |
| `DeepOKAN (2024)` | 20,792 | 0.043 | -17.754 | +0.897 | +0.680 | -39.6 | +0.048 | -0.088 | -0.700 | -103.4 | +0.131 |

---

## Benchmark 2: Equalized 15.5M-Parameter `MQAR` Extrapolation on RTX 3050 (`v0.6.0`)

Iso-architectural 6-layer mini-LLM (`d_model=384`, `n_heads=6`, `SwiGLU`, weight tying, `vocab_size=8192`), trained strictly on **`L_train = 128`** (550 steps, `AdamW`, 30-step warmup) and evaluated on blind extrapolation up to **`L = 4096` ($32\times$)**:

| Token Mixer | Params | Final Loss | `L=128` ($1\times$) | `L=256` ($2\times$) | `L=512` ($4\times$) | `L=1024` ($8\times$) | `L=2048` ($16\times$) | `L=4096` ($32\times$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`SOE-Abel (M=8)`** | 15.58M | **0.003** | **99.22%** | **98.44%** | **98.44%** | **85.94%** | **41.41%** | **14.84%** |
| `Hybrid 5:1 (SOE + RoPE)` | 15.57M | 0.035 | 98.44% | 97.66% | 82.03% | 17.97% | 5.47% | 7.03% |
| `Full Softmax + RoPE` | 15.55M | **0.003** | 97.66% | 89.84% | 56.25% | 16.41% | 9.38% | 4.69% |
| `Exp-Decay Linear` ($\lambda=0.015$) | 15.58M | 0.014 | 96.09% | 85.16% | 25.00% | 3.12% | 0.78% | 1.56% |

---

## Benchmark 3: RTX 3050 Hardware Scaling (`L = 512..32,768`, Forward + Backward)

| Seq. Len `L` | Method | Latency (ms) | Throughput (tok/s) | Peak VRAM (MB) |
| :---: | :--- | :---: | :---: | :---: |
| `8192` | `Dense O(L^2) Abel` vs **`SOE-Abel Chunk64`** | 2324.72 vs **511.72** (`4.54×` faster) | 3,524 vs **16,009** | 6,389.7 vs **674.1** (`9.5×` less) |
| `16384` | `Dense O(L^2) Abel` vs **`SOE-Abel Chunk64`** | `OOM` vs **1414.49** | --- vs **11,583** | `OOM` vs **1,329.0** |
| `32768` | `Dense O(L^2) Abel` vs **`SOE-Abel Chunk64`** | `OOM` vs **4527.13** | --- vs **7,238** | `OOM` vs **2,647.6** |

> **Golden Rule for `bfloat16`:** Accumulating `SOE` decay factors ($e^{-0.0002} \approx 0.9998 \to 1.0$ in 7-bit `bf16` mantissa) directly in `bfloat16` causes **116.7% tail error** at `L=2048`. Keeping state accumulation in `fp32` (`bf16-IO + fp32-acc`) cuts tail error by **492×** to **0.24%**.

---

## Quickstart

```bash
pip install -e .
```

### 1. Routh–Hurwitz Stability & Anchor-Residual Spectral Head
```python
import torch
from abelkan import MittagLefflerPade34, AnchorResidualMittagLefflerHead

pade = MittagLefflerPade34()
assert pade.certify_complex_stability()["all_hurwitz_stable"]  # max Re(z) = -1.3230 < 0

head = AnchorResidualMittagLefflerHead(d_model=36, out_channels=3, t_scale=48.0)
```

### 2. $O(L)$ Chunkwise `SOE-Abel` (`bf16-IO + fp32-acc`)
```python
import numpy as np
from abelkan import soe_chunkwise_forward

B, L, D, M = 1, 2048, 16, 8
q = np.abs(np.random.randn(B, L, D)) + 0.1
k = np.abs(np.random.randn(B, L, D)) + 0.1
v = np.random.randn(B, L, D)
dt = np.ones((B, L), dtype=np.float32)
lambdas = np.geomspace(1.0 / 4096.0, 0.5, M).astype(np.float32)
w = np.ones(M, dtype=np.float32) / M

out = soe_chunkwise_forward(q, k, v, dt, lambdas, w, chunk_size=64, use_bf16_io=True)
```

### 3. Run GPU Benchmarks (RTX 3050 / CUDA)
```bash
python benchmarks/gpu_bench_rtx3050.py
python benchmarks/gpu_mqar_15m_rtx3050.py
```

---

## Citation
If you use `abelkan` or `TerraKAN` in your research, please cite:
- **Author:** Ilya Kakhno (Faculty of Biology, Belarusian State University, ORCID: `0009-0005-5503-3269`)
- **Repository:** [https://github.com/Lolycut/TerraKAN](https://github.com/Lolycut/TerraKAN)
