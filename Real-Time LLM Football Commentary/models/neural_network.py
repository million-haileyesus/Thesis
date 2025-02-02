import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes, num_hidden_layers, dropout_rate, starting_size=2048):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_size = input_size

        sizes = [starting_size // (2 ** i) for i in range(num_hidden_layers)]

        for size in sizes:
            self.layers.extend([
                nn.Linear(prev_size, size),
                nn.BatchNorm1d(size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = size

        self.layers.append(nn.Linear(prev_size, num_classes))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
