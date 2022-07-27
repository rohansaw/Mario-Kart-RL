import torch
from torch import nn

a = torch.zeros([3, 3, 5, 6])
a = a.to("cuda:0")

h = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(), # max(0, x)
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(), # max(0, x)
    nn.Flatten(),
)
h = h.to("cuda:0")
o = torch.optim.SGD(h.parameters(), lr=0.01)

y = torch.sum(h(a))
y.backward()
o.step()

print(y)