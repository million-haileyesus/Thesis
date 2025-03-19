import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class GNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, num_classes, dropout_rate):
        super(GNN, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # First GCN layer
        self.layers.extend([
            GCNConv(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        
        # Hidden GCN layers
        for _ in range(num_hidden_layers - 1):
            self.layers.extend([
                GCNConv(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])

        # Final classification layer
        self.classifier = nn.Linear(hidden_size, num_classes)

        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Handle edge weights
        edge_weight = None
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_weight = data.edge_attr.view(-1).float()
            
        # Pass through layers
        for layer in self.layers:
            if isinstance(layer, GCNConv):
                x = layer(x, edge_index, edge_weight=edge_weight)
            else:
                x = layer(x)

        # Graph-level representation
        x = global_mean_pool(x, data.batch)
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)
