import torch
from abelkan import LogExpMultKAN

torch.manual_seed(0)
kan = LogExpMultKAN(in_features=3, num_branches=2)
opt = torch.optim.Adam(kan.parameters(), lr=0.05)

x = torch.rand(256, 3, dtype=torch.float64) * 0.8 + 0.1
y_true = (x[:, 1] ** 0.35) * (x[:, 2] ** 0.40)

for step in range(400):
    opt.zero_grad()
    loss = ((kan(x) - y_true) ** 2).mean() + 1e-3 * kan.l1_loss()
    loss.backward()
    opt.step()

print("Discovered symbolic formula:", kan.to_symbolic(["x0", "x1", "x2"], precision=3))
