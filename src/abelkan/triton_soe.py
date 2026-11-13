"""
GPU-ready Triton Chunkwise SOE-Abel Kernel Blueprint (abelkan v0.6.0).

Follows the Golden Rule for bfloat16 Tensor Core execution:
  1. Q, K, V inputs and output projections use bfloat16.
  2. Pade / SOE decay factors a_m = exp(-lambda_m * dt) and state accumulation S, Z
     MUST be computed and accumulated in float32 to prevent 7-bit mantissa collapse
     on slow power-law modes (exp(-lambda) ~ 0.996..0.9998).

When CUDA/Triton is unavailable (e.g. CPU environments), `soe_chunkwise_forward`
automatically dispatches to `abelkan.soe.soe_chunkwise_scan` in float32/float64.
"""

import importlib.util

import numpy as np

from .soe import soe_chunkwise_scan, to_bfloat16_sim

HAS_TRITON = importlib.util.find_spec("triton") is not None


def soe_chunkwise_forward(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    dt: np.ndarray,
    lambdas: np.ndarray,
    mode_weights: np.ndarray,
    chunk_size: int = 64,
    use_bf16_io: bool = True,
) -> np.ndarray:
    """
    Unified entry point: dispatches to Triton chunkwise kernel on CUDA+Triton
    when available, or falls back to exact fp32-accumulated CPU chunkwise scan.
    """
    if use_bf16_io:
        q_in = to_bfloat16_sim(q)
        k_in = to_bfloat16_sim(k)
        v_in = to_bfloat16_sim(v)
        out = soe_chunkwise_scan(
            q_in,
            k_in,
            v_in,
            dt.astype(np.float32),
            lambdas.astype(np.float32),
            mode_weights.astype(np.float32),
            chunk_size=chunk_size,
            acc_dtype=np.float32,
        )
        return to_bfloat16_sim(out)
    return soe_chunkwise_scan(
        q, k, v, dt, lambdas, mode_weights, chunk_size=chunk_size, acc_dtype=np.float64
    )
