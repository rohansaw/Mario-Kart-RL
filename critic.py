from torch.nn import Module
import torch.nn as nn

class SimpleCritic(Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = nn.functional.relu(self.fc1(x))
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out