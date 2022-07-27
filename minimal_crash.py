import torch
from torch import nn

a = torch.Tensor([3, 4])
a = a.to("cuda:0")
b = torch.nn.Linear(2, 10)
b = b.to("cuda:0")

print(b(a))
a = torch.zeros([3, 3, 5, 6])
a = a.to("cuda:0")
b = b.to("cuda:0")
# c = torch.nn.ReLU()(b(a))

# h = torch.nn.Sequential(
#     torch.nn.Conv2d(3, 10, 3),
#     torch.nn.ReLU(),
#     torch.nn.Flatten(),
#     torch.nn.Linear(120, 10),
#     torch.nn.Tanh(),
# )


h = nn.Sequential(
    # ShapePrintDebug(debug_interval=1, name="input"),
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(), # max(0, x)
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(), # max(0, x)
    # nn.Conv2d(3, 32, kernel_size=4, padding=1, stride=2),
    # nn.ReLU(), # max(0, x)
    # nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2),
    # nn.ReLU(), # max(0, x)
    nn.Flatten(),
)
h = h.to("cuda:0")
o = torch.optim.SGD(h.parameters(), lr=0.01)

y = torch.sum(h(a))
y.backward()
o.step()

print(y)