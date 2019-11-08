import torch
import torch.nn as nn

from encoders.pretrained_transformers.span_reprs import get_span_module


class SpanClassifier(nn.Module):
    def __init__(self, encoder, use_proj, proj_dim, hidden_dims, output_dim, 
            dropout_ratio=0.2, pooling_method='avg'):
        super(SpanClassifier, self).__init__()
        self.span_repr = get_span_module(
            method=pooling_method, 
            input_dim=encoder.hidden_size,
            use_proj=use_proj, 
            proj_dim=proj_dim
        )
        input_dim = self.span_repr.get_output_dim()
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
    
    def forward(self, encoded_input, start, end):
        span_repr = self.span_repr(encoded_input, start, end)
        return self.mlp(span_repr)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
