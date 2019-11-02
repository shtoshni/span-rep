import torch
import torch.nn as nn

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, 
            dropout_ratio=0.2):
        super(MultiLayerPerceptron, self).__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = list()
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.Dropout(dropout_ratio))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
