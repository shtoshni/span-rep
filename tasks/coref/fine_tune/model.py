import torch
import torch.nn as nn
from encoders.pretrained_transformers import Encoder
from encoders.pretrained_transformers.span_reprs import get_span_module


class CorefModel(nn.Module):
    def __init__(self, model='bert', model_size='base', just_last_layer=False,
                 span_dim=256, pool_method='avg', fine_tune=False, num_spans=1,
                 **kwargs):
        super(CorefModel, self).__init__()

        self.pool_method = pool_method
        self.num_spans = num_spans
        self.just_last_layer = just_last_layer
        self.encoder = Encoder(model=model, model_size=model_size, fine_tune=True,
                               cased=True)
        self.span_net = nn.ModuleDict()
        self.span_net['0'] = get_span_module(
            method=pool_method, input_dim=self.encoder.hidden_size,
            use_proj=True, proj_dim=span_dim)

        self.pooled_dim = self.span_net['0'].get_output_dim()

        self.label_net = nn.Sequential(
            nn.Linear(2 * self.pooled_dim, span_dim),
            nn.Tanh(),
            nn.LayerNorm(span_dim),
            nn.Dropout(0.2),
            nn.Linear(span_dim, 1),
            nn.Sigmoid()
        )

        self.training_criterion = nn.BCELoss()

    def get_core_params(self):
        return self.encoder.model.parameters()

    def calc_span_repr(self, encoded_input, span_indices, index='0'):
        span_start, span_end = span_indices[:, 0], span_indices[:, 1]
        span_repr = self.span_net[index](encoded_input, span_start, span_end)
        return span_repr

    def forward(self, batch_data):
        text, text_len = batch_data.text
        encoded_input = self.encoder(text.cuda(), just_last_layer=self.just_last_layer)

        s1_repr = self.calc_span_repr(encoded_input, batch_data.span1.cuda(), index='0')
        if self.num_spans > 1:
            s2_repr = self.calc_span_repr(encoded_input, batch_data.span2.cuda(), index='1')
        else:
            s2_repr = self.calc_span_repr(encoded_input, batch_data.span2.cuda(), index='0')

        pred_label = self.label_net(torch.cat([s1_repr, s2_repr], dim=-1))
        pred_label = torch.squeeze(pred_label, dim=-1)
        loss = self.training_criterion(pred_label, batch_data.label.cuda().float())
        if self.training:
            return loss
        else:
            return loss, pred_label
