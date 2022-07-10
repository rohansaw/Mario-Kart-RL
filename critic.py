from torch.nn import Module
import torch.nn as nn
from bitorch.layers import ShapePrintDebug
from bitorch.layers.config import config

# class SimpleCritic(Module):
#     def __init__(self, input_size):
#         super(SimpleCritic, self).__init__()
        
#         num_channels = input_size[2]
#         self.model = nn.Sequential(
#             nn.Conv3d(num_channels, 16, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(), # max(0, x)
#             nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(8960, 64),
#             # nn.Linear((input_size[0] // 4) * (input_size[1] // 4) * 32, 64),
#             # nn.Linear(128 * input_size[0] * input_size[1], 64),
#             nn.ReLU(), # max(0, x)
#             nn.Linear(64, 1),
#         )

#     def forward(self, x):
#         return self.model(x) # num actions x 1 


class SmolCritic(Module):
    def __init__(self, input_size):
        super(SmolCritic, self).__init__()
        
        num_channels = input_size[2]
        self.convolution = nn.Sequential(
            ShapePrintDebug(debug_interval=1, name="input"),
            nn.Conv2d(num_channels, 32, kernel_size=4, padding=1, stride=2),
            nn.ReLU(), # max(0, x)
            nn.Flatten(),
        )
        self.lstm1 = nn.LSTM(input_size=38400, hidden_size=64)
        self.classifier = nn.Sequential (
            nn.ReLU(), # max(0, x)
            nn.Linear(64, 1),
        )
        self.hidden = None

    def forward(self, x):
        # config.debug_activated = True
        convoluted = self.convolution(x)
        output, hidden = self.lstm1(convoluted)
        return self.classifier(hidden[0]) # num actions x 1 

    def reset_model(self):
        self.hidden = None

class BigCritic(Module):
    def __init__(self, input_size):
        super(BigCritic, self).__init__()
        
        num_channels = input_size[2]
        self.convolution = nn.Sequential(
            ShapePrintDebug(debug_interval=1, name="input"),
            nn.Conv2d(num_channels, 32, kernel_size=4, padding=1, stride=2),
            nn.ReLU(), # max(0, x)
            nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2),
            nn.ReLU(), # max(0, x)
            nn.Flatten(),
        )
        self.lstm1 = nn.LSTM(input_size=19200, hidden_size=512)
        self.classifier = nn.Sequential (
            nn.ReLU(), # max(0, x)
            nn.Linear(512, 128),
            nn.ReLU(), # max(0, x)
            nn.Linear(128, 1),
        )
        self.hidden = None

    def forward(self, x):
        # print(x)
        # config.debug_activated = True
        convoluted = self.convolution(x)
        # print(convoluted.shape)
        output, hidden = self.lstm1(convoluted)
        # print(output.shape, self.hidden[0].shape)
        return self.classifier(hidden[0]) # num actions x 1 

    def reset_model(self):
        self.hidden = None


# class Critic(torch.nn.Module):
#     def __init__(self, observation_space, hidden_layer_width=256):
#         super(Critic, self).__init__()
        
#         self.model = nn.Sequential(
#             nn.Linear(observation_space.shape[0], hidden_layer_width),
#             nn.ReLU(),
#             nn.Linear(hidden_layer_width, hidden_layer_width),
#             nn.ReLU(),
#             nn.Linear(hidden_layer_width, 1),
#         )


#     def forward(self, observation):
#         x = nn.ReLU()(self.l1(observation))
#         x = nn.ReLU()(self.l2(x))
#         return self.out(x)
