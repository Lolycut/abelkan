import torch

from abelkan import LogExpMultKAN, MittagLefflerSpectralHead, TemperatureTopKMoE


def test_logexp_multkan_symbolic_and_l1() -> None:
    torch.manual_seed(42)
    kan = LogExpMultKAN(in_features=3, num_branches=1).double()
    with torch.no_grad():
        kan.W_power.data[0, 0] = 0.0
        kan.W_power.data[0, 1] = 0.35
        kan.W_power.data[0, 2] = 0.40

    x = torch.rand(10, 3, dtype=torch.float64) * 0.8 + 0.1
    y = kan(x)
    y_exact = (x[:, 1] ** 0.35) * (x[:, 2] ** 0.40)
    assert torch.allclose(y, y_exact, atol=1e-6)
    sym = kan.to_symbolic(feature_names=["pH", "NO3", "W"], precision=2)
    assert "NO3^(0.35)" in sym and "W^(0.40)" in sym and "pH" not in sym
    assert kan.l1_loss().item() > 0.7


def test_spectral_head_and_moe() -> None:
    moe = TemperatureTopKMoE(d_model=16, num_experts=4, k=2, init_temp=1.5).double()
    x = torch.randn(4, 16, dtype=torch.float64)
    out, idx = moe(x)
    assert out.shape == (4, 16)
    assert idx.shape == (4, 2)
    assert torch.isfinite(moe.balance_loss())

    head = MittagLefflerSpectralHead(d_model=16, out_channels=3, num_modes=3).double()
    alpha = torch.full((4,), 0.72, dtype=torch.float64)
    t_eval = torch.linspace(0.0, 100.0, 25, dtype=torch.float64)
    traj = head(out, alpha, t_eval)
    assert traj.shape == (4, 25, 3)
    assert torch.isfinite(traj).all()
