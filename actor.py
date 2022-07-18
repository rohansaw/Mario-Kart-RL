from torch.nn import Module
import torch.nn as nn
import torch
from bitorch.layers import ShapePrintDebug, InputPrintDebug, WeightPrintDebug

from bitorch.layers.config import config

class SmolActor(Module):
    def __init__(self, input_size, output_size):
        super(SmolActor, self).__init__()
        
        num_channels = input_size[2]
        self.convolution = nn.Sequential(
            # ShapePrintDebug(debug_interval=1, name="input"),
            nn.Conv2d(num_channels, 32, kernel_size=4, padding=1, stride=2),
            nn.ReLU(), # max(0, x)
            nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2),
            nn.ReLU(), # max(0, x)
            nn.Flatten(),
        )
        self.lstm1 = nn.LSTM(input_size=19200, hidden_size=128, num_layers=2, dropout=0.2)
        self.classifier = nn.Sequential (
            nn.ReLU(), # max(0, x)
            nn.Linear(128, 32),
            nn.ReLU(), # max(0, x)
            nn.Linear(32, output_size),
            # ShapePrintDebug(debug_interval=1, name="softmax_input"),
            nn.Softmax(dim=0),
        )
        self.hidden = None

    def forward(self, x):
        # config.debug_activated = True
        convoluted = self.convolution(x)
        output, hidden = self.lstm1(convoluted)
        return self.classifier(hidden[0][-1]) # num actions x 1 

    def reset_model(self):
        self.hidden = None

class BigActor(Module):
    def __init__(self, input_size, output_size):
        super(BigActor, self).__init__()
        
        num_channels = input_size[2]
        self.convolution = nn.Sequential(
            ShapePrintDebug(debug_interval=1, name="input"),
            nn.Conv2d(num_channels, 32, kernel_size=4, padding=1, stride=2),
            nn.ReLU(), # max(0, x)
            nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2),
            nn.ReLU(), # max(0, x)
            nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2),
            nn.ReLU(), # max(0, x)
            nn.Flatten(),
        )
        self.lstm1 = nn.LSTM(input_size=8960, hidden_size=512)
        self.classifier = nn.Sequential (
            nn.ReLU(), # max(0, x)
            nn.Linear(512, 128),
            nn.ReLU(), # max(0, x)
            nn.Linear(128, output_size),
            nn.Softmax(),
        )
        self.hidden = None

    def forward(self, x):
        convoluted = self.convolution(x)
        output, hidden = self.lstm1(convoluted)
        return self.classifier(hidden[0]) # num actions x 1 

    def reset_model(self):
        self.hidden = None

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
