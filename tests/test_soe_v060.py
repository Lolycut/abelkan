import numpy as np
from scipy.optimize import nnls

from abelkan.soe import (
    blelloch_associative_scan,
    soe_associative_scan,
    soe_chunkwise_scan,
    soe_sequential_scan,
    to_bfloat16_sim,
)
from abelkan.triton_soe import soe_chunkwise_forward


def test_soe_kernel_approximation_m8():
    dt_grid = np.geomspace(1.0, 4096.0, 512)
    lambdas = np.geomspace(0.5 / 4096.0, 2.0, 8)
    A = np.exp(-dt_grid[:, None] * lambdas[None, :])
    y = (1.0 + dt_grid) ** (-0.5)
    weights = 1.0 / np.maximum(y, 1e-4)
    w, _ = nnls(A * weights[:, None], y * weights)
    l2_rel = float(np.linalg.norm(A @ w - y) / np.linalg.norm(y))
    assert l2_rel < 0.005, f"Expected L2 rel err < 0.005 for M=8, got {l2_rel}"


def test_blelloch_and_chunkwise_parity():
    rng = np.random.default_rng(123)
    B, L, D, M = 2, 128, 8, 6
    q = np.abs(rng.normal(0.5, 0.2, size=(B, L, D))) + 0.1
    k = np.abs(rng.normal(0.5, 0.2, size=(B, L, D))) + 0.1
    v = rng.normal(0.0, 1.0, size=(B, L, D))
    dt = np.ones((B, L), dtype=np.float64)
    lambdas = np.geomspace(1e-3, 0.5, M)
    w = np.ones(M) / M

    out_seq = soe_sequential_scan(q, k, v, dt, lambdas, w, acc_dtype=np.float64)
    out_assoc = soe_associative_scan(q, k, v, dt, lambdas, w, acc_dtype=np.float64)
    out_chunk = soe_chunkwise_scan(q, k, v, dt, lambdas, w, chunk_size=64, acc_dtype=np.float64)

    assert np.max(np.abs(out_seq - out_assoc)) < 1e-12
    assert np.max(np.abs(out_seq - out_chunk)) < 1e-12


def test_bf16_io_fp32_acc_stability():
    rng = np.random.default_rng(99)
    B, L, D, M = 1, 512, 8, 4
    q = np.abs(rng.normal(0.5, 0.2, size=(B, L, D))) + 0.1
    k = np.abs(rng.normal(0.5, 0.2, size=(B, L, D))) + 0.1
    v = rng.normal(0.0, 1.0, size=(B, L, D))
    dt = np.ones((B, L), dtype=np.float64)
    lambdas = np.geomspace(0.0002, 0.004, M)
    w = np.ones(M) / M

    out_ref = soe_sequential_scan(q, k, v, dt, lambdas, w, acc_dtype=np.float64)
    out_bf16 = soe_chunkwise_forward(q, k, v, dt, lambdas, w, chunk_size=64, use_bf16_io=True)
    rel_err = float(np.linalg.norm(out_bf16 - out_ref) / np.linalg.norm(out_ref))
    assert rel_err < 0.01, f"Expected bf16-IO + fp32-acc rel err < 0.01, got {rel_err}"


if __name__ == "__main__":
    test_soe_kernel_approximation_m8()
    test_blelloch_and_chunkwise_parity()
    test_bf16_io_fp32_acc_stability()
    print("All abelkan v0.6.0 SOE tests PASSED!")
