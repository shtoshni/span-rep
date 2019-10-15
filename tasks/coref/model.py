import torch
import torch.nn as nn
from encoders.pretrained_transformers import Encoder
from span_reprs import get_span_module


class CorefModel(nn.Module):
    def __init__(self, model='bert', model_size='base',
                 span_dim=256, pool_method='avg', fine_tune=False,
                 **kwargs):
        super(CorefModel, self).__init__()

        self.pool_method = pool_method
        self.encoder = Encoder(model=model, model_size=model_size, fine_tune=fine_tune,
                               cased=False)
        self.span_nn = get_span_module(method=pool_method, input_dim=self.encoder.hidden_size,
                                       use_proj=True, proj_dim=span_dim)
        self.pooled_dim = self.span_nn.get_output_dim()

        self.label_net = nn.Sequential(
            nn.Linear(2 * self.pooled_dim, span_dim),
            nn.Tanh(),
            nn.LayerNorm(span_dim),
            nn.Dropout(0.2),
            nn.Linear(span_dim, 1),
            nn.Sigmoid()
        )

        self.training_criterion = nn.BCELoss()

    def get_other_params(self):
        core_encoder_param_names = set()
        for name, param in self.encoder.model.named_parameters():
            if param.requires_grad:
                core_encoder_param_names.add(name)

        other_params = []
        print("\nParams outside core transformer params:\n")
        for name, param in self.named_parameters():
            if param.requires_grad and name not in core_encoder_param_names:
                print(name, param.data.size())
                other_params.append(param)
        print("\n")
        return other_params

    def get_core_params(self):
        return self.encoder.model.parameters()

    def calc_span_repr(self, encoded_input, span_indices):
        span_start, span_end = span_indices[:, 0], span_indices[:, 1]
        # if self.pool_method == "attn":
        #     span_repr = self.encoder.get_attn_span_repr(encoded_input, span_start, span_end)
        # elif self.pool_method == "coref":
        #     span_repr = self.encoder.get_coref_span_repr(encoded_input, span_start, span_end)
        # else:
        #     #span_repr = get_span_repr(encoded_input, span_start, span_end, method=self.pool_method)
        span_repr = self.span_nn(encoded_input, span_start, span_end)
        return span_repr

    def forward(self, batch_data):
        text, text_len = batch_data.text
        encoded_input = self.encoder(text.cuda())

        s1_repr = self.calc_span_repr(encoded_input, batch_data.span1.cuda())
        s2_repr = self.calc_span_repr(encoded_input, batch_data.span2.cuda())

        pred_label = self.label_net(torch.cat([s1_repr, s2_repr], dim=-1))
        pred_label = torch.squeeze(pred_label, dim=-1)
        loss = self.training_criterion(pred_label, batch_data.label.cuda().float())
        if self.training:
            return loss

        else:
            return loss, pred_label
