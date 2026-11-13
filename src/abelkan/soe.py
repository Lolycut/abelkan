import numpy as np


def to_bfloat16_sim(arr: np.ndarray) -> np.ndarray:
    """
    Exact IEEE/Google bfloat16 round-to-nearest-even simulation on float32 arrays.
    Truncates the 23-bit float32 mantissa to 7 bits (keeping sign + 8-bit exponent).
    """
    f32 = np.asarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    bf16_u32 = (u32 + rounding_bias) & np.uint32(0xFFFF0000)
    return bf16_u32.view(np.float32)


def soe_sequential_scan(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    dt: np.ndarray,
    lambdas: np.ndarray,
    mode_weights: np.ndarray,
    acc_dtype=np.float64,
) -> np.ndarray:
    """
    Reference sequential O(L) recurrent scan:
        S_{t,m} = a_{t,m} * S_{t-1,m} + u_{t,m}
    Args:
        q, k, v: (B, L, D)
        dt: (B, L) step sizes (dt[:, 0] ignored)
        lambdas: (M,) decay rates
        mode_weights: (B, L, M) or (M,)
    Returns:
        out: (B, L, D)
    """
    B, L, D = q.shape
    M = len(lambdas)
    q_a = q.astype(acc_dtype)
    k_a = k.astype(acc_dtype)
    v_a = v.astype(acc_dtype)
    dt_a = dt.astype(acc_dtype)
    lam_a = lambdas.astype(acc_dtype)

    S = np.zeros((B, M, D, D), dtype=acc_dtype)
    Z = np.zeros((B, M, D), dtype=acc_dtype)
    out = np.zeros((B, L, D), dtype=acc_dtype)

    for t in range(L):
        if t == 0:
            dec = np.ones((B, M), dtype=acc_dtype)
        else:
            dec = np.exp(-dt_a[:, t : t + 1] * lam_a[None, :])
            if acc_dtype == np.float32 and q.dtype == np.float32:
                pass
        kv = k_a[:, t, :, None] * v_a[:, t, None, :]  # (B, D, D)
        S = S * dec[:, :, None, None] + kv[:, None, :, :]
        Z = Z * dec[:, :, None] + k_a[:, t : t + 1, :]

        num_m = np.einsum("bd,bmde->bme", q_a[:, t, :], S)
        den_m = np.maximum(np.einsum("bd,bmd->bm", q_a[:, t, :], Z), 1e-6)[..., None]
        read_m = num_m / den_m
        w_t = mode_weights[:, t, :] if mode_weights.ndim == 3 else mode_weights[None, :]
        out[:, t, :] = np.sum(w_t[:, :, None] * read_m, axis=1)

    return out


def blelloch_associative_scan(a: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Parallel O(log L) Blelloch prefix scan for the affine associative operator:
        (a2, u2) o (a1, u1) = (a2 * a1, a2 * u1 + u2)
    Args:
        a: (..., L) decay multipliers (a[..., 0] unused or 1)
        u: (..., L, ...) updates
    Returns:
        s: (..., L, ...) prefix states S_t = a_t * S_{t-1} + u_t
    """
    L = a.shape[1]
    # Pad L to next power of 2 for clean tree reduction
    n_pad = 1 << (L - 1).bit_length()
    a_work = np.ones((a.shape[0], n_pad) + a.shape[2:], dtype=a.dtype)
    u_work = np.zeros((u.shape[0], n_pad) + u.shape[2:], dtype=u.dtype)
    a_work[:, :L] = a
    u_work[:, :L] = u

    # Hillis-Steele / Blelloch parallel-depth O(log L) doubling scan
    stride = 1
    while stride < n_pad:
        a_prev = a_work[:, :-stride]
        u_prev = u_work[:, :-stride]
        a_curr = a_work[:, stride:]
        u_curr = u_work[:, stride:]

        # Broadcast a_curr to u_curr shape
        a_curr_exp = a_curr.reshape(a_curr.shape + (1,) * (u_curr.ndim - a_curr.ndim))
        new_u = a_curr_exp * u_prev + u_curr
        new_a = a_curr * a_prev

        a_work[:, stride:] = new_a
        u_work[:, stride:] = new_u
        stride *= 2

    return u_work[:, :L]


def soe_associative_scan(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    dt: np.ndarray,
    lambdas: np.ndarray,
    mode_weights: np.ndarray,
    acc_dtype=np.float64,
) -> np.ndarray:
    """
    O(log L)-depth parallel associative scan implementation of SOE-Abel.
    """
    B, L, D = q.shape
    M = len(lambdas)
    q_a = q.astype(acc_dtype)
    k_a = k.astype(acc_dtype)
    v_a = v.astype(acc_dtype)
    dt_a = dt.astype(acc_dtype)
    lam_a = lambdas.astype(acc_dtype)

    dec = np.ones((B, L, M), dtype=acc_dtype)
    dec[:, 1:, :] = np.exp(-dt_a[:, 1:, None] * lam_a[None, None, :])

    kv = k_a[:, :, :, None] * v_a[:, :, None, :]  # (B, L, D, D)
    u_S = np.broadcast_to(kv[:, :, None, :, :], (B, L, M, D, D)).copy()
    u_Z = np.broadcast_to(k_a[:, :, None, :], (B, L, M, D)).copy()

    S_all = blelloch_associative_scan(dec, u_S)  # (B, L, M, D, D)
    Z_all = blelloch_associative_scan(dec, u_Z)  # (B, L, M, D)

    num_m = np.einsum("bld,blmde->blme", q_a, S_all)
    den_m = np.maximum(np.einsum("bld,blmd->blm", q_a, Z_all), 1e-6)[..., None]
    read_m = num_m / den_m
    w = mode_weights if mode_weights.ndim == 3 else mode_weights[None, None, :]
    return np.sum(w[..., None] * read_m, axis=2)


def soe_chunkwise_scan(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    dt: np.ndarray,
    lambdas: np.ndarray,
    mode_weights: np.ndarray,
    chunk_size: int = 64,
    acc_dtype=np.float64,
) -> np.ndarray:
    """
    Chunkwise Parallel SOE-Abel Scan (Triton-ready CPU/NumPy reference):
      - Intra-chunk (C x C): dense BLAS matmul (no Python loop over C tokens)
      - Inter-chunk (L / C steps): O(1) state transfer across chunks
    """
    B, L, D = q.shape
    M = len(lambdas)
    q_a = q.astype(acc_dtype)
    k_a = k.astype(acc_dtype)
    v_a = v.astype(acc_dtype)
    dt_a = dt.astype(acc_dtype)
    lam_a = lambdas.astype(acc_dtype)

    # Cumulative time from t=0 for exact intra-chunk decay factors
    dt_safe = dt_a.copy()
    dt_safe[:, 0] = 0.0
    t_cum = np.cumsum(dt_safe, axis=1)  # (B, L)

    out = np.zeros((B, L, D), dtype=acc_dtype)
    S_carry = np.zeros((B, M, D, D), dtype=acc_dtype)
    Z_carry = np.zeros((B, M, D), dtype=acc_dtype)

    for c_start in range(0, L, chunk_size):
        c_end = min(L, c_start + chunk_size)
        C = c_end - c_start
        qc = q_a[:, c_start:c_end, :]  # (B, C, D)
        kc = k_a[:, c_start:c_end, :]  # (B, C, D)
        vc = v_a[:, c_start:c_end, :]  # (B, C, D)
        tc = t_cum[:, c_start:c_end]  # (B, C)

        # Intra-chunk causal decay matrix: exp(-lam_m * (t_i - t_j)) for i >= j
        t_diff = np.maximum(tc[:, :, None] - tc[:, None, :], 0.0)  # (B, C, C)
        causal_tri = np.tril(np.ones((C, C), dtype=acc_dtype))[None, None, :, :]
        # (B, M, C, C)
        decay_intra = np.exp(-lam_a[None, :, None, None] * t_diff[:, None, :, :]) * causal_tri

        # Inter-chunk decay from chunk_start_prev to each t_i in chunk
        t_prev = t_cum[:, c_start - 1 : c_start] if c_start > 0 else tc[:, :1]
        decay_from_carry = np.exp(-lam_a[None, :, None] * np.maximum(tc[:, None, :] - t_prev[:, None, :], 0.0))  # (B, M, C)

        # Intra-chunk contribution via fast BLAS matmul
        qk_intra = np.einsum("bcd,bed->bce", qc, kc)[:, None, :, :] * decay_intra  # (B, M, C, C)
        num_intra = np.einsum("bmce,bed->bmcd", qk_intra, vc)  # (B, M, C, D)
        den_intra = np.sum(qk_intra, axis=-1)  # (B, M, C)

        # Carry contribution from previous chunks
        num_carry = np.einsum("bcd,bmde->bmce", qc, S_carry) * decay_from_carry[..., None]  # (B, M, C, D)
        den_carry = np.einsum("bcd,bmd->bmc", qc, Z_carry) * decay_from_carry  # (B, M, C)

        num_tot = num_intra + num_carry
        den_tot = np.maximum(den_intra + den_carry, 1e-6)[..., None]
        read_m = num_tot / den_tot  # (B, M, C, D)

        wc = mode_weights[:, c_start:c_end, :] if mode_weights.ndim == 3 else mode_weights[None, None, :]
        # wc: (B, C, M) -> transpose to (B, M, C, 1)
        wc_m = np.transpose(wc, (0, 2, 1))[..., None]
        out[:, c_start:c_end, :] = np.sum(wc_m * read_m, axis=1)

        # Update carry state to end of chunk (c_end - 1)
        t_end = tc[:, -1:]  # (B, 1)
        decay_end_carry = np.exp(-lam_a[None, :] * np.maximum(t_end - t_prev, 0.0))  # (B, M)
        decay_end_intra = np.exp(-lam_a[None, :, None] * np.maximum(t_end[:, None, :] - tc[:, None, :], 0.0))  # (B, M, C)

        kv_c = kc[:, :, :, None] * vc[:, :, None, :]  # (B, C, D, D)
        S_intra_end = np.einsum("bmc,bcde->bmde", decay_end_intra, kv_c)
        Z_intra_end = np.einsum("bmc,bcd->bmd", decay_end_intra, kc)

        S_carry = S_carry * decay_end_carry[:, :, None, None] + S_intra_end
        Z_carry = Z_carry * decay_end_carry[:, :, None] + Z_intra_end

    return out
