import torch

from abelkan import AbelCausalAttention, AbelTransformerLayer


def test_strict_causality() -> None:
    torch.manual_seed(0)
    attn = AbelCausalAttention(d_model=16, decay_mode="abel")
    x = torch.randn(2, 10, 16, dtype=torch.float64)
    ts = torch.linspace(0.0, 90.0, 10, dtype=torch.float64).unsqueeze(0).expand(2, 10)

    out1, a1, _ = attn(x, ts)

    x_pert = x.clone()
    ts_pert = ts.clone()
    x_pert[:, 7:, :] += 5.0
    ts_pert[:, 7:] += 100.0
    out2, a2, _ = attn(x_pert, ts_pert)

    assert torch.allclose(out1[:, :7, :], out2[:, :7, :], atol=1e-12)
    assert torch.allclose(a1[:, :7], a2[:, :7], atol=1e-12)


def test_decay_modes_and_mask() -> None:
    x = torch.randn(2, 6, 16, dtype=torch.float64)
    ts = torch.tensor([[0.0, 5.0, 10.0, 20.0, 50.0, 100.0]] * 2, dtype=torch.float64)
    mask = torch.tensor([[1, 1, 1, 0, 1, 1], [1, 0, 1, 1, 1, 0]], dtype=torch.float64)

    for mode in ("abel", "exp_alibi", "none"):
        layer = AbelTransformerLayer(d_model=16, decay_mode=mode)
        out, a_hat, s_hat = layer(x, ts, attn_mask=mask)
        assert out.shape == (2, 6, 16)
        assert a_hat.shape == (2, 6)
        assert s_hat.shape == (2, 6)
        assert torch.isfinite(out).all()
