"""
GPU Benchmark for abelkan v0.6.0 (Part 2 Stage G) — Tuned for Laptop RTX 3050 (4-6 GB VRAM).

Measures Forward + Backward Latency (ms), Throughput (tokens/s), and Peak VRAM (MB)
across L in {512, 2048, 8192, 16384, 32768} with graceful OOM handling for O(L^2) baselines.

Usage (on your RTX 3050 laptop with PyTorch CUDA):
    python gpu_bench_rtx3050.py
"""

import argparse
import gc
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkwiseSOEAbelCUDA(nn.Module):
    """
    Pure PyTorch CUDA Chunkwise SOE-Abel Attention (no Triton required -> works on Windows & Linux).
    Golden Rule for RTX 3050 bfloat16:
      - Q, K, V projections run in bfloat16 (Tensor Cores).
      - Exponential decays exp(-lambda_m * dt) and state S, Z accumulate strictly in float32.
      - Intra-chunk (C x C) uses batched matmul; inter-chunk carries O(1) state across L/C steps.
    """

    def __init__(self, d_model: int = 384, n_heads: int = 6, M: int = 8, chunk_size: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.M = M
        self.chunk_size = chunk_size

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.alpha_gate = nn.Linear(d_model, n_heads * 2)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        lams = torch.logspace(math.log10(1.0 / 4096.0), math.log10(0.5), M, dtype=torch.float32)
        self.register_buffer("lambdas", lams)
        self.mode_logits = nn.Parameter(torch.linspace(2.0, -2.0, M, dtype=torch.float32).unsqueeze(0).repeat(n_heads, 1))
        self.register_buffer("slow_bias", torch.linspace(1.5, 0.0, M, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, Dh, M, C = self.n_heads, self.head_dim, self.M, self.chunk_size

        qkv = self.qkv(x).view(B, L, 3, H, Dh)
        # Positive feature map in float32 for numerical stability in denominator
        q = F.softplus(qkv[:, :, 0].float()) + 1e-3  # (B, L, H, Dh)
        k = F.softplus(qkv[:, :, 1].float()) + 1e-3
        v = qkv[:, :, 2].float()

        ag = self.alpha_gate(x).float().view(B, L, H, 2)
        alpha_t = 0.35 + 0.57 * torch.sigmoid(ag[..., 0])  # (B, L, H)
        beta_t = torch.sigmoid(ag[..., 1] + 1.5)  # (B, L, H)

        lams = self.lambdas.to(device=x.device, dtype=torch.float32)  # (M,)
        S_carry = torch.zeros((B, H, M, Dh, Dh), device=x.device, dtype=torch.float32)
        Z_carry = torch.zeros((B, H, M, Dh), device=x.device, dtype=torch.float32)

        outs = []
        for c0 in range(0, L, C):
            c1 = min(L, c0 + C)
            clen = c1 - c0
            qc = q[:, c0:c1] * 1.0  # (B, clen, H, Dh)
            kc = k[:, c0:c1] * beta_t[:, c0:c1].unsqueeze(-1)
            vc = v[:, c0:c1]

            # Transpose to (B, H, clen, Dh)
            qh = qc.permute(0, 2, 1, 3)
            kh = kc.permute(0, 2, 1, 3)
            vh = vc.permute(0, 2, 1, 3)

            pos = torch.arange(clen, device=x.device, dtype=torch.float32)
            dt_intra = torch.clamp(pos.unsqueeze(1) - pos.unsqueeze(0), min=0.0)  # (clen, clen)
            causal = torch.tril(torch.ones((clen, clen), device=x.device, dtype=torch.float32))
            # (1, 1, M, clen, clen)
            dec_intra = torch.exp(-lams.view(1, 1, M, 1, 1) * dt_intra.view(1, 1, 1, clen, clen)) * causal.view(1, 1, 1, clen, clen)

            qk = torch.matmul(qh, kh.transpose(-2, -1))  # (B, H, clen, clen)
            qk_m = qk.unsqueeze(2) * dec_intra  # (B, H, M, clen, clen)
            num_intra = torch.matmul(qk_m, vh.unsqueeze(2))  # (B, H, M, clen, Dh)
            den_intra = qk_m.sum(dim=-1)  # (B, H, M, clen)

            # Carry from previous chunks
            dec_carry = torch.exp(-lams.view(1, 1, M, 1) * (pos + 1.0).view(1, 1, 1, clen))  # (1, 1, M, clen)
            num_carry = torch.einsum("bhcd,bhmde->bhmce", qh, S_carry) * dec_carry.unsqueeze(-1)
            den_carry = torch.einsum("bhcd,bhmd->bhmc", qh, Z_carry) * dec_carry

            read_m = (num_intra + num_carry) / torch.clamp(den_intra + den_carry, min=1e-6).unsqueeze(-1)  # (B, H, M, clen, Dh)

            # Alpha-modulated mode mixture
            a_chunk = (1.0 - alpha_t[:, c0:c1]).permute(0, 2, 1).unsqueeze(-1)  # (B, H, clen, 1)
            logits_m = self.mode_logits.view(1, H, 1, M) + a_chunk * self.slow_bias.view(1, 1, 1, M)
            w_m = torch.softmax(logits_m, dim=-1).permute(0, 1, 3, 2).unsqueeze(-1)  # (B, H, M, clen, 1)
            out_c = (w_m * read_m).sum(dim=2).permute(0, 2, 1, 3).reshape(B, clen, D)
            outs.append(out_c)

            # Update carry state to end of chunk
            dec_end_carry = torch.exp(-lams.view(1, 1, M) * float(clen))
            dec_end_intra = torch.exp(-lams.view(1, 1, M, 1) * (float(clen - 1) - pos).view(1, 1, 1, clen))
            kv_c = kh.unsqueeze(-1) * vh.unsqueeze(-2)  # (B, H, clen, Dh, Dh)
            S_intra = torch.einsum("bhmc,bhcde->bhmde", dec_end_intra.expand(B, H, M, clen), kv_c)
            Z_intra = torch.einsum("bhmc,bhcd->bhmd", dec_end_intra.expand(B, H, M, clen), kh)

            S_carry = S_carry * dec_end_carry.unsqueeze(-1).unsqueeze(-1) + S_intra
            Z_carry = Z_carry * dec_end_carry.unsqueeze(-1) + Z_intra

        out = torch.cat(outs, dim=1).to(x.dtype)
        return self.out_proj(out)


class DenseAbelBaseline(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 6):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.raw_gamma = nn.Parameter(torch.full((n_heads,), 0.85))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, Dh = self.n_heads, self.head_dim
        qkv = self.qkv(x).view(B, L, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(Dh))
        pos = torch.arange(L, device=x.device, dtype=x.dtype)
        dt = torch.clamp(pos.unsqueeze(1) - pos.unsqueeze(0), min=0.0)
        bias = -F.softplus(self.raw_gamma).view(1, H, 1, 1) * 0.4 * torch.log1p(dt).view(1, 1, L, L)
        causal = torch.tril(torch.ones((L, L), device=x.device, dtype=torch.bool))
        scores = scores + bias
        scores = scores.masked_fill(~causal.view(1, 1, L, L), -1e4)
        w = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        return self.out_proj(torch.matmul(w, v).permute(0, 2, 1, 3).reshape(B, L, D))


class SDPABaseline(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 6):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, Dh = self.n_heads, self.head_dim
        qkv = self.qkv(x).view(B, L, 3, H, Dh).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)
        return self.out_proj(out.permute(0, 2, 1, 3).reshape(B, L, D))


def benchmark_one(fn, L: int, d_model: int = 384, device: str = "cuda", dtype=torch.bfloat16, warmup: int = 2, iters: int = 5):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = fn().to(device=device, dtype=dtype)
    x = torch.randn((1, L, d_model), device=device, dtype=dtype, requires_grad=True)

    # Warmup
    for _ in range(warmup):
        y = model(x)
        y.sum().backward()
        model.zero_grad(set_to_none=True)
        x.grad = None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = model(x)
        y.sum().backward()
        model.zero_grad(set_to_none=True)
        x.grad = None
    torch.cuda.synchronize()
    ms = ((time.perf_counter() - t0) / iters) * 1000.0
    vram_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
    tok_s = L / (ms / 1000.0)
    del model, x, y
    gc.collect()
    torch.cuda.empty_cache()
    return ms, tok_s, vram_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 2048, 8192, 16384, 32768])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected! Run this script on your RTX 3050 laptop.")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"GPU: {gpu_name} ({vram_gb:.2f} GB VRAM) | dtype={dtype} + fp32 SOE acc")
    print(f"{'L':>6s} | {'Method':24s} | {'Lat (ms)':>9s} | {'Tok/s':>10s} | {'Peak VRAM (MB)':>14s}")
    print("-" * 76)

    methods = [
        ("PyTorch SDPA", lambda: SDPABaseline(args.d_model, args.n_heads)),
        ("Dense O(L^2) Abel", lambda: DenseAbelBaseline(args.d_model, args.n_heads)),
        ("SOE-Abel Chunk64 (v0.6)", lambda: ChunkwiseSOEAbelCUDA(args.d_model, args.n_heads, M=8, chunk_size=64)),
    ]

    for L in args.lengths:
        for name, mfn in methods:
            try:
                ms, tok_s, vram = benchmark_one(mfn, L, args.d_model, "cuda", dtype)
                print(f"{L:6d} | {name:24s} | {ms:9.2f} | {tok_s:10.0f} | {vram:14.1f}")
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"{L:6d} | {name:24s} | {'OOM':>9s} | {'---':>10s} | {'OOM (>VRAM)':>14s}")
        print("-" * 76)


if __name__ == "__main__":
    main()
