from torch.nn import Module
import torch.nn as nn
import torch

class SimpleActor(Module):
    def __init__(self, input_size, output_size):
        super(SimpleActor, self).__init__()
        
        num_channels = input_size[2]
        self.model = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=4, padding=0, stride=2),
            nn.ReLU(), # max(0, x)
            nn.Conv2d(64, 128, kernel_size=4, padding=0, stride=2),
            nn.ReLU(), # max(0, x)
            nn.Conv2d(128, 256, kernel_size=4, padding=0, stride=2),
            nn.ReLU(), # max(0, x)
            nn.Flatten(),
            nn.Linear(272384, 512),
            # nn.Linear(4 * input_size[0] * input_size[1], 512),
            nn.ReLU(), # max(0, x)
            nn.Linear(512, 128),
            nn.Sigmoid(), # max(0, x)
            nn.Linear(128, output_size),
            nn.Softmax(),
        )

    def forward(self, x):
        # print(x)
        return self.model(x) # num actions x 1 

# ToDo, not working yet
class LSTMActor(Module):
    def __init__(self, input_size, hidden_size, num_layers, num_actions):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_actions = num_actions
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, inputs, hidden):
        """
        inputs: (batch_size, seq_len, input_size)
        hidden: (num_layers, batch_size, hidden_size)
        """
        output, hidden = self.lstm(inputs, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

    def select_actions(self, observations: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        observations: (batch_size, seq_len, input_size)
        """
        output, hidden = self.forward(observations, hidden)
        return output, hidden