from torch.nn import Module
import torch.nn as nn
import torch
from bitorch.layers import ShapePrintDebug, InputPrintDebug, WeightPrintDebug

from bitorch.layers.config import config

class SimpleActor(Module):
    def __init__(self, input_size, output_size):
        super(SimpleActor, self).__init__()
        
        print("input size:", input_size)
        num_channels = input_size[2]
        self.model = nn.Sequential(
            # InputPrintDebug(debug_interval=1, num_outputs=1),
            # ShapePrintDebug(debug_interval=1, name="input"),
            nn.Conv2d(num_channels, 64, kernel_size=4, padding=1, stride=2),
            # WeightPrintDebug(module=nn.Conv2d(num_channels, 64, kernel_size=4, padding=1, stride=2), name="weight 1", debug_interval=1),
            # ShapePrintDebug(debug_interval=1, name="1"),
            # InputPrintDebug(debug_interval=1, num_outputs=1, name="1"),
            nn.ReLU(), # max(0, x)
            # nn.BatchNorm2d(64),
            WeightPrintDebug(module=nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2), name="weight 2", debug_interval=1),
            # ShapePrintDebug(debug_interval=1, name="2"),
            # InputPrintDebug(debug_interval=1, num_outputs=1, name="2"),
            nn.ReLU(), # max(0, x)
            # nn.BatchNorm2d(128),
            # nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2),
            # InputPrintDebug(debug_interval=1, num_outputs=1, name="3"),
            # nn.ReLU(), # max(0, x)
            # nn.BatchNorm2d(256),
            nn.Flatten(),
            # InputPrintDebug(debug_interval=1, num_outputs=1, name="4"),
            nn.Linear(((input_size[0] // 4) * (input_size[1] // 4) * 128), 512),
            # nn.Linear(((input_size[0] // 8) * (input_size[1] // 8) * 256), 512),
            # nn.Linear(4 * input_size[0] * input_size[1], 512),
            nn.ReLU(), # max(0, x)
            # nn.BatchNorm2d(512),
            # InputPrintDebug(debug_interval=1, num_outputs=1, name="5"),
            # nn.Linear(512, 128),
            # nn.Sigmoid(), # max(0, x)
            # nn.BatchNorm2d(128),
            InputPrintDebug(debug_interval=1, num_outputs=1, name="6"),
            WeightPrintDebug(debug_interval=1, num_outputs=1, module=nn.Linear(512, output_size), name="weight 6"),
            # nn.Linear(512, output_size),
            nn.Softmax(),
        )
        # self.model = nn.Sequential(
        #         ShapePrintDebug(debug_interval=1, name="1"),
        #         nn.Conv2d(num_channels, 64, kernel_size=5),
        #         # nn.BatchNorm2d(64),
        #         nn.Tanh(),
        #         nn.MaxPool2d(2, 2),

        #         nn.Conv2d(64, 64, kernel_size=5),
        #         # nn.BatchNorm2d(64),
        #         nn.Tanh(),
        #         nn.MaxPool2d(2, 2),
        #         ShapePrintDebug(debug_interval=1, name="3"),

        #         nn.Flatten(),

        #         nn.Linear(64 * 4 * 7, 1000),
        #         # nn.BatchNorm1d(1000),
        #         nn.Tanh(),

        #         InputPrintDebug(debug_interval=1, num_outputs=1, name="6"),
        #         nn.Linear(1000, output_size),
        #         nn.Softmax(),
        #     )

        print("first linear layer:", (input_size[0] * input_size[1]))

    def forward(self, x):
        # print(x)
        config.debug_activated = True
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
