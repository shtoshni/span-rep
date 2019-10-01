import torch
import torch.nn as nn
from encoder import Encoder


class Net(nn.Module):
    def __init__(self, model='bert', model_size='base', top_rnns=False,
                 vocab_size=None, device='cpu', finetuning=False):
        super().__init__()
        self.encoder = Encoder(model=model, model_size=model_size,
                               fine_tune=finetuning)

        self.top_rnns = top_rnns
        hidden_size = self.encoder.hidden_size
        if top_rnns:
            self.rnn = nn.LSTM(
                bidirectional=True, num_layers=2,
                input_size=hidden_size, hidden_size=hidden_size//2, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.device = device
        self.finetuning = finetuning

    def forward(self, x, y, ):
        '''
        x: (N, T). int64
        y: (N, T). int64

        Returns
        enc: (N, T, VOCAB)
        '''
        x = x.to(self.device)
        y = y.to(self.device)

        if self.training and self.finetuning:
            # print("->bert.train()")
            self.encoder.train()
            enc = self.encoder(x, just_last_layer=True)
        else:
            self.encoder.eval()
            with torch.no_grad():
                enc = self.encoder(x, just_last_layer=True)

        if self.top_rnns:
            enc, _ = self.rnn(enc)
        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat

    def print_model_info(self):
        """Prints model parameters and their total count"""
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                dims = list(param.data.size())
                local_params = 1
                for dim in dims:
                    local_params *= dim
                total_params += local_params
                print(name, param.data.size())
        print("\nTotal Params:{:.2f} (in millions)".format(total_params/10**6))
