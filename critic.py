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


class SimpleCritic(Module):
    def __init__(self, input_size):
        super(SimpleCritic, self).__init__()
        
        print("input size:", input_size)
        num_channels = input_size[2]
        self.convolution = nn.Sequential(
            # InputPrintDebug(debug_interval=1, num_outputs=1),
            ShapePrintDebug(debug_interval=1, name="input"),
            nn.Conv2d(num_channels, 32, kernel_size=4, padding=1, stride=2),
            # WeightPrintDebug(module=nn.Conv2d(num_channels, 64, kernel_size=4, padding=1, stride=2), name="weight 1", debug_interval=1),
            # ShapePrintDebug(debug_interval=1, name="1"),
            # InputPrintDebug(debug_interval=1, num_outputs=1, name="1"),
            nn.ReLU(), # max(0, x)
            nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2),
            # WeightPrintDebug(module=nn.Conv2d(num_channels, 64, kernel_size=4, padding=1, stride=2), name="weight 1", debug_interval=1),
            # ShapePrintDebug(debug_interval=1, name="1"),
            # InputPrintDebug(debug_interval=1, num_outputs=1, name="1"),
            nn.ReLU(), # max(0, x)
            # nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2),
            # InputPrintDebug(debug_interval=1, num_outputs=1, name="3"),
            # nn.ReLU(), # max(0, x)
            # nn.BatchNorm2d(256),
            nn.Flatten(),
        )
            # InputPrintDebug(debug_interval=1, num_outputs=1, name="4"),
        self.lstm1 = nn.LSTM(input_size=19200, hidden_size=512)
            # nn.Linear(((input_size[0] // 4) * (input_size[1] // 4) * 128), 512),
            # nn.Linear(((input_size[0] // 8) * (input_size[1] // 8) * 256), 512),
            # nn.Linear(4 * input_size[0] * input_size[1], 512),
        self.classifier = nn.Sequential (
            nn.ReLU(), # max(0, x)
            # nn.BatchNorm2d(512),
            # InputPrintDebug(debug_interval=1, num_outputs=1, name="5"),
            nn.Linear(512, 128),
            nn.ReLU(), # max(0, x)
            # nn.Sigmoid(), # max(0, x)
            # nn.BatchNorm2d(128),
            # InputPrintDebug(debug_interval=1, num_outputs=1, name="6"),
        # self.lstm2 = nn.LSTM(128, 64),
            nn.Linear(128, 1),
            # WeightPrintDebug(debug_interval=1, num_outputs=1, module=nn.Linear(512, output_size), name="weight 6"),
            # nn.Linear(512, output_size),
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
        self.hidden = None
        print("first linear layer:", (input_size[0] * input_size[1]))

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
