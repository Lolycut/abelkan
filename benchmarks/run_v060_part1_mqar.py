import gc
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath("intermediate/pylib"))
sys.path.insert(0, os.path.abspath("output/abelkan/src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIR = "output"

# Vocabulary layout (vocab_size = 24):
#   0..7   : Keys (8 distinct keys)
#   8..15  : Values (8 distinct values)
#   16..23 : Noise tokens (8 distinct noise tokens)
VOCAB_SIZE = 24
NUM_KEYS = 8
VAL_OFFSET = 8
NOISE_START = 16
NOISE_END = 24


def generate_mqar_batch(rng, batch_size: int, seq_len: int, num_pairs: int = 3, burst_delay: bool = False):
    tokens = rng.integers(NOISE_START, NOISE_END, size=(batch_size, seq_len), dtype=np.int64)
    query_pos = np.zeros((batch_size, num_pairs), dtype=np.int64)
    target_vals = np.zeros((batch_size, num_pairs), dtype=np.int64)

    for b in range(batch_size):
        keys = rng.choice(NUM_KEYS, size=num_pairs, replace=False)
        vals = rng.integers(VAL_OFFSET, VAL_OFFSET + NUM_KEYS, size=num_pairs)

        if burst_delay:
            prefix_end = max(2 * num_pairs + 2, int(0.15 * seq_len))
            q_start = max(prefix_end + 4, int(0.90 * seq_len))
            kv_starts = np.sort(rng.choice(np.arange(0, prefix_end - 1, 2), size=num_pairs, replace=False))
            q_positions = np.sort(rng.choice(np.arange(q_start, seq_len), size=num_pairs, replace=False))
        else:
            prefix_end = max(2 * num_pairs + 2, int(0.65 * seq_len))
            q_start = max(prefix_end + 2, seq_len - num_pairs - max(2, int(0.15 * seq_len)))
            kv_starts = np.sort(rng.choice(np.arange(0, prefix_end - 1, 2), size=num_pairs, replace=False))
            q_positions = np.sort(rng.choice(np.arange(q_start, seq_len), size=num_pairs, replace=False))

        for i in range(num_pairs):
            tokens[b, kv_starts[i]] = keys[i]
            tokens[b, kv_starts[i] + 1] = vals[i]

        perm = rng.permutation(num_pairs)
        for i, p_idx in enumerate(perm):
            tokens[b, q_positions[i]] = keys[p_idx]
            query_pos[b, i] = q_positions[i]
            target_vals[b, i] = vals[p_idx]

    return tokens, query_pos, target_vals


def generate_overwrite_batch(rng, batch_size: int, seq_len: int = 48, num_pairs: int = 3):
    tokens = rng.integers(NOISE_START, NOISE_END, size=(batch_size, seq_len), dtype=np.int64)
    query_pos = np.zeros((batch_size, 1), dtype=np.int64)
    new_vals = np.zeros((batch_size, 1), dtype=np.int64)
    old_vals = np.zeros((batch_size, 1), dtype=np.int64)

    for b in range(batch_size):
        keys = rng.choice(NUM_KEYS, size=num_pairs, replace=False)
        target_k = keys[0]
        v_old, v_new = rng.choice(np.arange(VAL_OFFSET, VAL_OFFSET + NUM_KEYS), size=2, replace=False)
        tokens[b, 0] = target_k
        tokens[b, 1] = v_old
        pos = 4
        for j in range(1, num_pairs):
            tokens[b, pos] = keys[j]
            tokens[b, pos + 1] = rng.integers(VAL_OFFSET, VAL_OFFSET + NUM_KEYS)
            pos += 3
        ov_pos = int(0.65 * seq_len)
        tokens[b, ov_pos] = target_k
        tokens[b, ov_pos + 1] = v_new
        tokens[b, seq_len - 1] = target_k
        query_pos[b, 0] = seq_len - 1
        new_vals[b, 0] = v_new
        old_vals[b, 0] = v_old

    return tokens, query_pos, new_vals, old_vals


def apply_rope(x):
    B, H, L, Dh = x.shape
    half = Dh // 2
    pos = np.arange(L, dtype=np.float64)[:, None]
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, half, dtype=np.float64) / half))[None, :]
    ang = pos * inv_freq
    cos_t = torch.tensor(np.cos(ang), dtype=x.dtype)[None, None, :, :]
    sin_t = torch.tensor(np.sin(ang), dtype=x.dtype)[None, None, :, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos_t - x2 * sin_t, x1 * sin_t + x2 * cos_t], dim=-1)


class RoPECausalAttention(nn.Module):
    def __init__(self, d_model=32, n_heads=2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q = apply_rope(qkv[:, :, 0].transpose(1, 2))
        k = apply_rope(qkv[:, :, 1].transpose(1, 2))
        v = qkv[:, :, 2].transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        idx = torch.arange(L).unsqueeze(0)
        cmask = (idx.unsqueeze(2) >= idx.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        neg_inf = torch.tensor(-1e9, dtype=x.dtype)
        scores = torch.where(cmask, scores, neg_inf)
        w = torch.softmax(scores, dim=-1)
        out = torch.matmul(w, v).transpose(1, 2).reshape(B, L, D)
        return self.out(out)


class ExpDecayLinearMixer(nn.Module):
    def __init__(self, d_model=32, n_heads=2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.raw_gamma = nn.Parameter(torch.full((n_heads,), -1.5, dtype=torch.float64))

    def forward(self, x):
        B, L, D = x.shape
        H, Dh = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, L, 3, H, Dh)
        q = F.softplus(qkv[:, :, 0]) + 1e-3
        k = F.softplus(qkv[:, :, 1]) + 1e-3
        v = qkv[:, :, 2]
        decay = torch.exp(-F.softplus(self.raw_gamma)).reshape(1, H)

        S = torch.zeros((B, H, Dh, Dh), dtype=x.dtype)
        Z = torch.zeros((B, H, Dh), dtype=x.dtype)
        outs = []
        for t in range(L):
            qt, kt, vt = q[:, t], k[:, t], v[:, t]
            S = S * decay.unsqueeze(-1).unsqueeze(-1) + kt.unsqueeze(-1) * vt.unsqueeze(-2)
            Z = Z * decay.unsqueeze(-1) + kt
            num = torch.matmul(qt.unsqueeze(2), S).squeeze(2)
            den = torch.clamp(torch.sum(qt * Z, dim=-1), min=1e-6).unsqueeze(-1)
            outs.append((num / den).reshape(B, D))
        return self.out(torch.stack(outs, dim=1))


class SOEAbelLinearMixer(nn.Module):
    def __init__(self, d_model=32, n_heads=2, M=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.M = M
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.alpha_gate = nn.Linear(d_model, n_heads * 2)
        self.out = nn.Linear(d_model, d_model)
        lams = np.geomspace(1.0 / 1024.0, 1.0, M).astype(np.float64)
        self.lambdas = torch.tensor(lams, dtype=torch.float64)
        self.mode_logits = nn.Parameter(torch.zeros((n_heads, M), dtype=torch.float64))

    def forward(self, x):
        B, L, D = x.shape
        H, Dh, M = self.n_heads, self.head_dim, self.M
        qkv = self.qkv(x).reshape(B, L, 3, H, Dh)
        q = F.softplus(qkv[:, :, 0]) + 1e-3
        k = F.softplus(qkv[:, :, 1]) + 1e-3
        v = qkv[:, :, 2]

        ag = self.alpha_gate(x).reshape(B, L, H, 2)
        alpha_t = 0.35 + 0.57 * torch.sigmoid(ag[..., 0])
        beta_t = torch.sigmoid(ag[..., 1] + 1.5)

        decays = torch.exp(-self.lambdas).reshape(1, 1, M)
        slow_bias = torch.linspace(1.5, 0.0, M, dtype=x.dtype).reshape(1, 1, M)

        S = torch.zeros((B, H, M, Dh, Dh), dtype=x.dtype)
        Z = torch.zeros((B, H, M, Dh), dtype=x.dtype)
        outs = []
        for t in range(L):
            qt, kt, vt = q[:, t], k[:, t], v[:, t]
            bt = beta_t[:, t].unsqueeze(-1)
            kv = kt.unsqueeze(-1) * vt.unsqueeze(-2)
            S = S * decays.unsqueeze(-1).unsqueeze(-1) + (bt.unsqueeze(-1).unsqueeze(-1) * kv.unsqueeze(2))
            Z = Z * decays.unsqueeze(-1) + (bt.unsqueeze(-1) * kt.unsqueeze(2))

            num_m = torch.matmul(qt.unsqueeze(2).unsqueeze(3), S).squeeze(3)
            den_m = torch.clamp(torch.sum(qt.unsqueeze(2) * Z, dim=-1), min=1e-6).unsqueeze(-1)
            read_m = num_m / den_m

            w_t = torch.softmax(self.mode_logits.unsqueeze(0) + (1.0 - alpha_t[:, t]).unsqueeze(-1) * slow_bias, dim=-1)
            out_t = torch.sum(w_t.unsqueeze(-1) * read_m, dim=2).reshape(B, D)
            outs.append(out_t)
        return self.out(torch.stack(outs, dim=1))


class MiniLMBlock(nn.Module):
    def __init__(self, d_model=32, n_heads=2, mixer_type="rope"):
        super().__init__()
        if mixer_type == "rope":
            self.mixer = RoPECausalAttention(d_model, n_heads)
        elif mixer_type == "exp":
            self.mixer = ExpDecayLinearMixer(d_model, n_heads)
        elif mixer_type == "soe_abel":
            self.mixer = SOEAbelLinearMixer(d_model, n_heads, M=8)
        else:
            raise ValueError(mixer_type)
        self.shift_proj = nn.Linear(2 * d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x_prev = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)
        x_mix = x + 0.5 * torch.tanh(self.shift_proj(torch.cat([x, x_prev], dim=-1)))
        h = self.norm1(x + self.mixer(x_mix))
        return self.norm2(h + self.ffn(h))


class IsoMiniLM(nn.Module):
    def __init__(self, mixer_schedule, vocab_size=VOCAB_SIZE, d_model=32, n_heads=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Linear(vocab_size, d_model, bias=False)
        self.blocks = nn.ModuleList([MiniLMBlock(d_model, n_heads, m) for m in mixer_schedule])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens_np):
        oh = np.eye(self.vocab_size, dtype=np.float64)[tokens_np]
        h = self.tok_emb(torch.tensor(oh, dtype=torch.float64))
        for blk in self.blocks:
            h = blk(h)
        return self.lm_head(self.norm_f(h))


def train_and_eval_lm(mixer_schedule, seed: int, steps: int = 240, batch_size: int = 16, l_train: int = 24):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = IsoMiniLM(mixer_schedule, vocab_size=VOCAB_SIZE, d_model=32, n_heads=2)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=7e-3)

    t0 = time.perf_counter()
    for step in range(steps):
        if step % 5 == 4:
            toks, qpos, tvals, _ = generate_overwrite_batch(rng, batch_size, seq_len=l_train, num_pairs=3)
        else:
            toks, qpos, tvals = generate_mqar_batch(rng, batch_size, seq_len=l_train, num_pairs=3, burst_delay=False)

        logits = model(toks)
        B, K = qpos.shape
        q_logits = torch.stack([logits[bi, int(qpos[bi, ki]), :] for bi in range(B) for ki in range(K)], dim=0)
        y_oh_t = torch.tensor(np.eye(VOCAB_SIZE, dtype=np.float64)[tvals.reshape(-1)], dtype=torch.float64)
        mx = q_logits.max(dim=-1, keepdim=True)[0]
        log_p = q_logits - mx - torch.log(torch.sum(torch.exp(q_logits - mx), dim=-1, keepdim=True))
        loss = -torch.mean(torch.sum(y_oh_t * log_p, dim=-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 15 == 0:
            gc.collect()
    train_s = time.perf_counter() - t0

    # Test 1A: Blind length extrapolation L = 24..384 (16x L_train)
    mqar_acc = {}
    for L_test in [24, 48, 96, 192, 384]:
        toks, qpos, tvals = generate_mqar_batch(rng, 32, seq_len=L_test, num_pairs=3, burst_delay=False)
        with torch.no_grad():
            logits_np = model(toks).numpy()
        preds = np.array([np.argmax(logits_np[bi, qpos[bi]], axis=-1) for bi in range(32)])
        mqar_acc[str(L_test)] = round(float(np.mean(preds == tvals) * 100.0), 2)

    # Test 1B: Burst-Delay (75% pure noise gap) at L = 128, 256
    burst_acc = {}
    for L_test in [128, 256]:
        toks, qpos, tvals = generate_mqar_batch(rng, 32, seq_len=L_test, num_pairs=3, burst_delay=True)
        with torch.no_grad():
            logits_np = model(toks).numpy()
        preds = np.array([np.argmax(logits_np[bi, qpos[bi]], axis=-1) for bi in range(32)])
        burst_acc[str(L_test)] = round(float(np.mean(preds == tvals) * 100.0), 2)

    # Test 1C: Overwrite at L = 48 (2x L_train)
    toks, qpos, new_v, old_v = generate_overwrite_batch(rng, 48, seq_len=48, num_pairs=3)
    with torch.no_grad():
        logits_np = model(toks).numpy()
    preds = np.array([np.argmax(logits_np[bi, qpos[bi]], axis=-1) for bi in range(48)])
    ov_new = round(float(np.mean(preds == new_v) * 100.0), 2)
    ov_old = round(float(np.mean(preds == old_v) * 100.0), 2)

    return {
        "params": n_params,
        "train_s": round(train_s, 2),
        "mqar_acc": mqar_acc,
        "burst_acc": burst_acc,
        "overwrite_new_acc": ov_new,
        "overwrite_stale_acc": ov_old,
    }


def main():
    CONFIGS = [
        ("Full Softmax + RoPE", ["rope", "rope"]),
        ("Exp-Decay Linear", ["exp", "exp"]),
        ("SOE-Abel (M=8)", ["soe_abel", "soe_abel"]),
        ("Hybrid 1:1 (SOE-Abel + RoPE)", ["soe_abel", "rope"]),
    ]
    SEEDS = [42, 101, 777]
    results = {}
    for name, sched in CONFIGS:
        print(f"\n=== Training {name} (3 seeds) ===")
        runs = [train_and_eval_lm(sched, sd, steps=240, batch_size=16, l_train=24) for sd in SEEDS]
        agg = {
            "params": runs[0]["params"],
            "mqar_mean": {L: round(float(np.mean([r["mqar_acc"][L] for r in runs])), 2) for L in ["24", "48", "96", "192", "384"]},
            "burst_mean": {L: round(float(np.mean([r["burst_acc"][L] for r in runs])), 2) for L in ["128", "256"]},
            "overwrite_new_mean": round(float(np.mean([r["overwrite_new_acc"] for r in runs])), 2),
            "overwrite_stale_mean": round(float(np.mean([r["overwrite_stale_acc"] for r in runs])), 2),
        }
        results[name] = agg
        print(f"  Params={agg['params']} | MQAR L=24->384: {agg['mqar_mean']} | Burst(128,256): {agg['burst_mean']} | Overwrite(new/stale): {agg['overwrite_new_mean']}% / {agg['overwrite_stale_mean']}%")

    with open(os.path.join(OUT_DIR, "v060_part1_mqar.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved output/v060_part1_mqar.json!")


if __name__ == "__main__":
    main()
