"""Vector-latent controller; names match the original NES sample state dictionaries."""

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class Controller(nn.Module):
    def __init__(self, input_size: int, layer_size: int, output_size: int, hidden_layers: int = 0):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Linear(input_size, layer_size)
        if hidden_layers:
            self.hidden_layers_list = nn.ModuleList(
                nn.Linear(layer_size, layer_size) for _ in range(hidden_layers)
            )
        self.output_layer = nn.Linear(layer_size, output_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        if self.hidden_layers:
            for layer in self.hidden_layers_list:
                x = torch.relu(layer(x))
        return torch.softmax(self.output_layer(x), dim=-1)

    def vector(self):
        return parameters_to_vector(self.parameters()).detach().clone()

    def set_vector(self, vector):
        if vector.numel() != sum(parameter.numel() for parameter in self.parameters()):
            raise ValueError("Controller vector has an incompatible length")
        with torch.no_grad():
            vector_to_parameters(vector.detach().clone(), self.parameters())
