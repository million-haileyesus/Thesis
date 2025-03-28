import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes, num_hidden_layers, hidden_size, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        
        self.layers.extend([
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        
        for _ in range(num_hidden_layers - 1):
            self.layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])

        self.layers.append(nn.Linear(hidden_size, num_classes))

        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
