import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath("output/abelkan/src"))

import numpy as np
from abelkan.soe import (
    soe_sequential_scan,
    soe_associative_scan,
    soe_chunkwise_scan,
    to_bfloat16_sim,
)

OUT_DIR = "output"


def run_stage_b():
    print("=== Part 2 Stage B: Sequential vs Blelloch O(log L) vs Chunkwise (C=64) ===")
    rng = np.random.default_rng(42)
    M = 8
    D = 16
    B = 2
    lambdas = np.geomspace(1.0 / 2048.0, 0.5, M)
    w = np.ones(M) / M

    scan_results = {}
    for L in [64, 256, 1024, 2048, 4096]:
        q = np.abs(rng.normal(0.5, 0.3, size=(B, L, D))) + 0.1
        k = np.abs(rng.normal(0.5, 0.3, size=(B, L, D))) + 0.1
        v = rng.normal(0.0, 1.0, size=(B, L, D))
        dt = np.ones((B, L), dtype=np.float64)

        t0 = time.perf_counter()
        out_seq = soe_sequential_scan(q, k, v, dt, lambdas, w, acc_dtype=np.float64)
        t_seq_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        out_chunk = soe_chunkwise_scan(q, k, v, dt, lambdas, w, chunk_size=64, acc_dtype=np.float64)
        t_chunk_ms = (time.perf_counter() - t0) * 1000.0

        diff_chunk = float(np.max(np.abs(out_seq - out_chunk)))

        if L <= 1024:
            t0 = time.perf_counter()
            out_assoc = soe_associative_scan(q, k, v, dt, lambdas, w, acc_dtype=np.float64)
            t_assoc_ms = (time.perf_counter() - t0) * 1000.0
            diff_assoc = float(np.max(np.abs(out_seq - out_assoc)))
        else:
            t_assoc_ms = None
            diff_assoc = None

        speedup = t_seq_ms / max(t_chunk_ms, 1e-6)
        scan_results[str(L)] = {
            "seq_ms": round(t_seq_ms, 2),
            "chunk64_ms": round(t_chunk_ms, 2),
            "assoc_ms": round(t_assoc_ms, 2) if t_assoc_ms is not None else None,
            "chunk_vs_seq_speedup": round(speedup, 2),
            "max_abs_diff_chunk": diff_chunk,
            "max_abs_diff_assoc": diff_assoc,
        }
        print(
            f"  L={L:4d} | Seq={t_seq_ms:7.2f} ms | Chunk64={t_chunk_ms:6.2f} ms ({speedup:5.2f}x) | "
            f"MaxDiff(Chunk)={diff_chunk:.2e}" + (f" | MaxDiff(Blelloch)={diff_assoc:.2e}" if diff_assoc is not None else "")
        )
    return scan_results


def run_stage_c():
    print("\n=== Part 2 Stage C: bfloat16 Precision Trap vs fp32-acc Golden Rule (L=2048) ===")
    rng = np.random.default_rng(777)
    B, L, D, M = 2, 2048, 16, 8
    # Slow power-law tail modes: exp(-lam) in [0.996, 0.9998]
    lambdas = np.geomspace(0.0002, 0.004, M)
    w = np.ones(M, dtype=np.float64) / M

    q64 = np.abs(rng.normal(0.5, 0.25, size=(B, L, D))) + 0.1
    k64 = np.abs(rng.normal(0.5, 0.25, size=(B, L, D))) + 0.1
    v64 = rng.normal(0.0, 1.0, size=(B, L, D))
    dt64 = np.ones((B, L), dtype=np.float64)

    # 1. Exact float64 reference
    out_fp64 = soe_sequential_scan(q64, k64, v64, dt64, lambdas, w, acc_dtype=np.float64)

    # 2. Trap: Naive bf16-all (inputs AND state accumulation S, Z, decay in bfloat16)
    q_bf = to_bfloat16_sim(q64)
    k_bf = to_bfloat16_sim(k64)
    v_bf = to_bfloat16_sim(v64)
    dec_bf = to_bfloat16_sim(np.exp(-lambdas))  # Rounded in 7-bit mantissa!

    S_bf = np.zeros((B, M, D, D), dtype=np.float32)
    Z_bf = np.zeros((B, M, D), dtype=np.float32)
    out_bf16_all = np.zeros((B, L, D), dtype=np.float32)
    for t in range(L):
        dec_t = np.ones((B, M), dtype=np.float32) if t == 0 else dec_bf[None, :]
        kv = to_bfloat16_sim(k_bf[:, t, :, None] * v_bf[:, t, None, :])
        S_bf = to_bfloat16_sim(to_bfloat16_sim(S_bf * dec_t[:, :, None, None]) + kv[:, None, :, :])
        Z_bf = to_bfloat16_sim(to_bfloat16_sim(Z_bf * dec_t[:, :, None]) + k_bf[:, t : t + 1, :])
        num_m = to_bfloat16_sim(np.einsum("bd,bmde->bme", q_bf[:, t, :], S_bf))
        den_m = np.maximum(to_bfloat16_sim(np.einsum("bd,bmd->bm", q_bf[:, t, :], Z_bf)), 1e-6)[..., None]
        out_bf16_all[:, t, :] = to_bfloat16_sim(np.sum((w[None, :, None].astype(np.float32)) * (num_m / den_m), axis=1))

    # 3. Golden Rule: bf16 I/O + fp32 state accumulation (abelkan v0.6.0)
    out_fp32_acc = soe_chunkwise_scan(q_bf, k_bf, v_bf, dt64.astype(np.float32), lambdas.astype(np.float32), w.astype(np.float32), chunk_size=64, acc_dtype=np.float32)
    out_bf16_io_fp32_acc = to_bfloat16_sim(out_fp32_acc)

    def metrics(pred):
        rel_l2 = float(np.linalg.norm(pred - out_fp64) / np.linalg.norm(out_fp64))
        max_abs = float(np.max(np.abs(pred - out_fp64)))
        # Tail error on last 256 tokens (where long-range memory lives)
        tail_rel = float(np.linalg.norm(pred[:, -256:] - out_fp64[:, -256:]) / np.linalg.norm(out_fp64[:, -256:]))
        return {"rel_l2_err": round(rel_l2, 6), "tail_256_rel_err": round(tail_rel, 6), "max_abs_err": round(max_abs, 6)}

    m_trap = metrics(out_bf16_all)
    m_gold = metrics(out_bf16_io_fp32_acc)
    # Also check decay rounding at exp(-0.0002) = 0.9998
    dec_exact = float(np.exp(-0.0002))
    dec_rounded = float(to_bfloat16_sim(np.array([dec_exact], dtype=np.float32))[0])

    print(f"  Decay exp(-0.0002) = {dec_exact:.6f} -> bf16 rounds to {dec_rounded:.6f} (forgetting vanishes!)")
    print(f"  Naive bf16-all (TRAP)       : RelL2={m_trap['rel_l2_err']:.4f} | Tail256_RelL2={m_trap['tail_256_rel_err']:.4f} | MaxAbs={m_trap['max_abs_err']:.4f}")
    print(f"  bf16-IO + fp32-acc (v0.6.0) : RelL2={m_gold['rel_l2_err']:.6f} | Tail256_RelL2={m_gold['tail_256_rel_err']:.6f} | MaxAbs={m_gold['max_abs_err']:.6f}")

    return {
        "slow_decay_exact": round(dec_exact, 6),
        "slow_decay_bf16_rounded": round(dec_rounded, 6),
        "naive_bf16_all": m_trap,
        "bf16_io_fp32_acc": m_gold,
    }


def main():
    sb = run_stage_b()
    sc = run_stage_c()
    with open(os.path.join(OUT_DIR, "v060_stageBC_scan_bf16.json"), "w", encoding="utf-8") as f:
        json.dump({"stage_b_scan_parity": sb, "stage_c_bf16_trap": sc}, f, indent=2)
    print("\nSaved output/v060_stageBC_scan_bf16.json!")


if __name__ == "__main__":
    main()
