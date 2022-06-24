from torch.nn import Module
import torch.nn as nn

class SimpleCritic(Module):
    def __init__(self, input_size):
        super(SimpleCritic, self).__init__()
        
        num_channels = input_size[2]
        self.model = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(), # max(0, x)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9830400, 64),
            # nn.Linear(128 * input_size[0] * input_size[1], 64),
            nn.ReLU(), # max(0, x)
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x) # num actions x 1 


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
