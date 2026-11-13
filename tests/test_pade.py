import numpy as np
import torch
from scipy.special import erfcx

from abelkan import MittagLefflerPade34


def test_pade_matches_erfcx_at_alpha_05() -> None:
    ml = MittagLefflerPade34()
    z = torch.logspace(-3, 3, 200, dtype=torch.float64)
    alpha = torch.tensor(0.50, dtype=torch.float64)
    pred = ml(z, alpha).detach().numpy()
    exact = erfcx(z.numpy())
    max_err = float(np.max(np.abs(pred - exact)))
    assert max_err < 5e-5, f"max_err={max_err}"


def test_pade_gradcheck_including_alpha_05() -> None:
    ml = MittagLefflerPade34()
    z = torch.tensor([0.1, 1.5, 12.0], dtype=torch.float64, requires_grad=True)
    alpha = torch.tensor(0.50, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(ml, (z, alpha), eps=1e-6, atol=1e-4)


def test_c1_bridge_at_alpha_090() -> None:
    ml = MittagLefflerPade34(bridge_start=0.90)
    z = torch.logspace(-2, 2, 32, dtype=torch.float64)
    eps = 1e-6
    v_left = ml(z, torch.tensor(0.90 - eps, dtype=torch.float64))
    v_mid = ml(z, torch.tensor(0.90, dtype=torch.float64))
    v_right = ml(z, torch.tensor(0.90 + eps, dtype=torch.float64))
    assert torch.allclose(v_left, v_mid, atol=1e-4)
    assert torch.allclose(v_right, v_mid, atol=1e-4)


def test_descartes_certify_pole_free() -> None:
    ml = MittagLefflerPade34()
    cert = ml.certify_pole_free()
    assert cert["certified_pole_free"] is True
    assert cert["min_q1"] > 0.0 and cert["min_q4"] > 0.0


def test_float32_compatibility() -> None:
    ml = MittagLefflerPade34()
    z32 = torch.logspace(-2, 2, 50, dtype=torch.float32)
    a32 = torch.tensor(0.65, dtype=torch.float32)
    out = ml(z32, a32)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_routh_hurwitz_complex_stability() -> None:
    ml = MittagLefflerPade34()
    cert = ml.certify_complex_stability()
    assert cert["certified_hurwitz_stable"] is True
    assert cert["min_routh_d2"] > 0.0 and cert["min_routh_d3"] > 0.0
