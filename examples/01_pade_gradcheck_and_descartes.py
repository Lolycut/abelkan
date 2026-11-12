import torch
from abelkan import MittagLefflerPade34

ml = MittagLefflerPade34()
cert = ml.certify_pole_free()
print("Descartes Pole-Freedom Certificate on [0, +inf):", cert)

z = torch.tensor([0.2, 1.5, 10.0], dtype=torch.float64, requires_grad=True)
alpha = torch.tensor(0.50, dtype=torch.float64, requires_grad=True)
ok = torch.autograd.gradcheck(ml, (z, alpha), eps=1e-6, atol=1e-4)
print("Autograd gradcheck at alpha=0.50:", ok)
