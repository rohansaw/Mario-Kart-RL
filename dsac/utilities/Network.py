import torch


class Network(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, output_activation=torch.nn.Identity()):
        super(Network, self).__init__()
        self.layer_1 = torch.nn.Conv2d(input_dimension[2], 32, kernel_size=4, padding=1, stride=2)
        self.layer_2 = torch.nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2)
        self.output_layer = torch.nn.Linear(307200, output_dimension)
        self.output_activation = output_activation

    def forward(self, inpt):
        layer_1_output = torch.nn.functional.relu(self.layer_1(inpt))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_2_output = torch.flatten(layer_2_output)
        output = self.output_activation(self.output_layer(layer_2_output))
        return output
