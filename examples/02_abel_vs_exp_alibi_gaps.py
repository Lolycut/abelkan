import torch
from abelkan import AbelTransformerLayer

timestamps = torch.tensor([[0.0, 7.0, 14.0, 21.0, 120.0]], dtype=torch.float64)
x = torch.randn(1, 5, 32, dtype=torch.float64)

for mode in ("abel", "exp_alibi", "none"):
    layer = AbelTransformerLayer(d_model=32, decay_mode=mode).double()
    out, alpha_hat, sigma = layer(x, timestamps)
    a_v = alpha_hat[0, -1].item()
    s_v = sigma[0, -1].item()
    print(f"mode={mode:9s} | out_norm={out.norm().item():.3f} | alpha={a_v:.3f} +/- {s_v:.3f}")
