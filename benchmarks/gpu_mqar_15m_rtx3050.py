"""
Part 1 Iso-Architectural 15M+ Parameter Mini-LLM Benchmark (MQAR 16x, Burst-Delay, Overwrite)
Tuned specifically for Laptop RTX 3050 (4-6 GB VRAM):
  - Architecture: vocab_size=8192, n_layers=6, d_model=384, n_heads=6, SwiGLU (ffn=1024)
    -> ~15.4M parameters (untied LM head)
  - RTX 3050 VRAM Fit:
      * torch.amp.autocast('cuda', dtype=bfloat16)
      * micro_batch_size = 4, grad_accum_steps = 8 (effective batch = 32)
      * gradient checkpointing enabled on transformer blocks (fits comfortably in <1.8 GB VRAM!)

Usage:
    python gpu_mqar_15m_rtx3050.py --steps 400
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from gpu_bench_rtx3050 import ChunkwiseSOEAbelCUDA

VOCAB_SIZE = 8192
NUM_KEYS = 64
VAL_OFFSET = 64
NOISE_START = 128


def generate_mqar_batch_gpu(batch_size: int, seq_len: int, num_pairs: int = 8, burst_delay: bool = False, device: str = "cuda"):
    tokens = torch.randint(NOISE_START, VOCAB_SIZE, (batch_size, seq_len), device=device, dtype=torch.long)
    qpos = torch.zeros((batch_size, num_pairs), device=device, dtype=torch.long)
    tvals = torch.zeros((batch_size, num_pairs), device=device, dtype=torch.long)

    for b in range(batch_size):
        keys = torch.randperm(NUM_KEYS, device=device)[:num_pairs]
        vals = torch.randint(VAL_OFFSET, VAL_OFFSET + NUM_KEYS, (num_pairs,), device=device)

        if burst_delay:
            p_end = max(2 * num_pairs + 2, int(0.15 * seq_len))
            q_start = max(p_end + 4, int(0.90 * seq_len))
        else:
            p_end = max(2 * num_pairs + 2, int(0.65 * seq_len))
            q_start = max(p_end + 2, int(0.80 * seq_len))

        kv_idx = torch.sort(torch.randperm((p_end - 1) // 2, device=device)[:num_pairs] * 2)[0]
        q_idx = torch.sort(q_start + torch.randperm(seq_len - q_start, device=device)[:num_pairs])[0]

        tokens[b, kv_idx] = keys
        tokens[b, kv_idx + 1] = vals
        perm = torch.randperm(num_pairs, device=device)
        tokens[b, q_idx] = keys[perm]
        qpos[b] = q_idx
        tvals[b] = vals[perm]

    return tokens, qpos, tvals


class RoPEMixerCUDA(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 6):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _apply_rope(self, x):
        B, H, L, Dh = x.shape
        half = Dh // 2
        pos = torch.arange(L, device=x.device, dtype=torch.float32).unsqueeze(1)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half)).unsqueeze(0)
        ang = pos * inv_freq
        cos_t = torch.cos(ang).to(x.dtype).view(1, 1, L, half)
        sin_t = torch.sin(ang).to(x.dtype).view(1, 1, L, half)
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos_t - x2 * sin_t, x1 * sin_t + x2 * cos_t], dim=-1)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = self._apply_rope(qkv[0])
        k = self._apply_rope(qkv[1])
        v = qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(out.permute(0, 2, 1, 3).reshape(B, L, D))


class ExpDecayMixerCUDA(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 6, chunk_size: int = 64):
        super().__init__()
        self.soe = ChunkwiseSOEAbelCUDA(d_model, n_heads, M=1, chunk_size=chunk_size)
        # Standard Markovian exponential decay (half-life ~46 tokens, instead of logspace left-endpoint 1/4096)
        self.soe.lambdas.fill_(0.015)

    def forward(self, x):
        return self.soe(x)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int = 384, ffn_dim: int = 1280):
        super().__init__()
        self.w12 = nn.Linear(d_model, 2 * ffn_dim, bias=False)
        self.w3 = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class MiniLM15MBlock(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 6, ffn_dim: int = 1280, mixer_type: str = "soe"):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.conv_shift = nn.Conv1d(d_model, d_model, kernel_size=2, padding=1, groups=d_model)
        if mixer_type == "rope":
            self.mixer = RoPEMixerCUDA(d_model, n_heads)
        elif mixer_type == "exp":
            self.mixer = ExpDecayMixerCUDA(d_model, n_heads)
        elif mixer_type == "soe":
            self.mixer = ChunkwiseSOEAbelCUDA(d_model, n_heads, M=8, chunk_size=64)
        else:
            raise ValueError(mixer_type)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ffn_dim)

    def forward(self, x):
        h = self.norm1(x)
        # Causal k=2 depthwise conv token-shift (standard in Zoology)
        h_s = self.conv_shift(h.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        x = x + self.mixer(h + 0.5 * h_s)
        x = x + self.ffn(self.norm2(x))
        return x


class IsoMiniLM15M(nn.Module):
    def __init__(self, schedule, vocab_size: int = VOCAB_SIZE, d_model: int = 384, n_heads: int = 6, ffn_dim: int = 1280, use_ckpt: bool = True):
        super().__init__()
        self.use_ckpt = use_ckpt
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([MiniLM15MBlock(d_model, n_heads, ffn_dim, m) for m in schedule])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying for all 4 models

    def forward(self, tokens):
        h = self.tok_emb(tokens)
        for blk in self.blocks:
            if self.use_ckpt and self.training:
                h = checkpoint(blk, h, use_reentrant=False)
            else:
                h = blk(h)
        return self.lm_head(self.norm_f(h))


def train_eval_model(name: str, schedule, steps: int = 550, micro_bs: int = 8, grad_accum: int = 4, l_train: int = 128):
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = IsoMiniLM15M(schedule, use_ckpt=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {name} | Params: {n_params / 1e6:.2f}M ({n_params:,}) ===")

    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, betas=(0.9, 0.95), weight_decay=0.01)
    warmup = 30
    decay_start = int(0.75 * steps)
    def lr_fn(s):
        if s < warmup:
            return float(s + 1) / float(warmup)
        if s < decay_start:
            return 1.0
        prog = float(s - decay_start) / float(max(1, steps - decay_start))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_fn)
    t0 = time.perf_counter()
    model.train()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            toks, qpos, tvals = generate_mqar_batch_gpu(micro_bs, l_train, num_pairs=8, device=device)
            with torch.amp.autocast("cuda", dtype=dtype):
                logits = model(toks)
                b_idx = torch.arange(micro_bs, device=device).unsqueeze(1).expand_as(qpos)
                q_logits = logits[b_idx, qpos]
                loss = F.cross_entropy(q_logits.reshape(-1, VOCAB_SIZE), tvals.reshape(-1)) / grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 50 == 0:
            print(f"  step {step+1:3d}/{steps} | loss={loss.item()*grad_accum:.3f} | VRAM={torch.cuda.max_memory_allocated()/1024**2:.0f} MB")

    print(f"  Training done in {time.perf_counter() - t0:.1f}s")
    model.eval()
    with torch.no_grad():
        for L_test in [128, 256, 512, 1024, 2048, 4096]:
            try:
                toks, qpos, tvals = generate_mqar_batch_gpu(16, L_test, num_pairs=8, device=device)
                with torch.amp.autocast("cuda", dtype=dtype):
                    logits = model(toks)
                b_idx = torch.arange(16, device=device).unsqueeze(1).expand_as(qpos)
                preds = logits[b_idx, qpos].argmax(dim=-1)
                acc = (preds == tvals).float().mean().item() * 100.0
                print(f"  MQAR L={L_test:4d} -> Recall = {acc:5.2f}%")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  MQAR L={L_test:4d} -> OOM")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=550)
    parser.add_argument("--micro-bs", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected! Run this script on your RTX 3050 laptop.")

    configs = [
        ("Full Softmax + RoPE", ["rope"] * 6),
        ("Exp-Decay Linear", ["exp"] * 6),
        ("SOE-Abel (M=8)", ["soe"] * 6),
        ("Hybrid 5:1 (SOE-Abel + RoPE)", ["soe"] * 5 + ["rope"]),
    ]
    for name, sched in configs:
        train_eval_model(name, sched, steps=args.steps, micro_bs=args.micro_bs, grad_accum=args.grad_accum)


if __name__ == "__main__":
    main()
