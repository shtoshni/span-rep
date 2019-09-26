import torch
import torch.nn as nn
import logging

from transformers import *

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

MODEL_LIST = ['bert', 'roberta', 'gpt2']
# MODEL_LIST = ['bert', 'spanbert', ''roberta', 'gpt2']
BERT_MODEL_SIZES = ['base', 'large']
GPT2_MODEL_SIZES = ['', 'medium', 'large']


class Encoder(nn.Module):
    def __init__(self, model='bert', model_type='base', cased=True,
                 fine_tune=False):
        super(Encoder, self).__init__()
        assert(model in MODEL_LIST)

        self.model = None
        self.tokenizer = None
        self.num_layers = None
        self.hidden_size = None

        # First initialize the model and tokenizer
        model_name = ''
        # Model is one of the BERT variants
        if 'bert' in model:
            assert (model_type in BERT_MODEL_SIZES)
            model_name = model + "-" + model_type
            if model == 'bert' and not cased:
                # Only original BERT supports uncased models
                model_name += '-uncased'
            elif model == 'roberta':
                # RoBERTa model types have no casing suffix in HuggingFace map
                # So we don't modify the model name
                pass
            else:
                model_name += '-cased'

            print (model_name)
            if model == 'bert':
                self.model = BertModel.from_pretrained(model_name)
                self.tokenizer = BertTokenizer.from_pretrained(model_name)
            elif model == 'roberta':
                self.model = RobertaModel.from_pretrained(model_name)
                self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            # elif model == 'spanbert':

            self.num_layers = self.model.config.num_hidden_layers
            self.hidden_size = self.model.config.hidden_size

        elif model == 'gpt2':
            assert (model_type in GPT2_MODEL_SIZES)
            model_name = model
            if model_type:
                model_name += "-" + model_type

            self.model = GPT2Model.from_pretrained(model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.num_layers = self.model.config.n_layer
            self.hidden_size = self.model.config.n_embd

        # Set requires_grad to False if not fine tuning
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.config.output_hidden_states = True
        # Set parameters required on top of pre-trained models
        self.weighing_params = nn.Parameter(torch.ones(self.num_layers))

        # Attention-based Span representation parameters - MIGHT NOT BE USED
        self.attention_weight = nn.Parameter(torch.ones(self.hidden_size))


    def tokenize_input(self, sentence, max_length=512):
        """
        sentence: a whole string containing all the tokens (NOT A LIST).
        """
        return torch.tensor(self.tokenizer.encode(sentence, max_length=max_length,
            add_special_tokens=True)).unsqueeze(dim=0).cuda()

    def encode_tokens(self, batch_ids):
        """
        Encode a batch of token IDs with a learned
        batch_ids: B x L
        """
        input_mask = (batch_ids > 0).cuda().float()
        print (self.model.config.output_hidden_states)
        outputs = self.model(
            batch_ids, attention_mask=input_mask)  # B x L x E
        print (outputs)
        print (len(outputs))
        print (outputs[1].shape)
        print (outputs[0].shape)

        # Encoded layers also has the embedding layer - 0th entry
        # print (len(encoded_layers))
        encoded_layers = encoded_layers[1:]

        wtd_encoded_repr = 0
        soft_weight = nn.functional.softmax(self.weighing_params, dim=0)

        for i in range(self.num_layers):
            wtd_encoded_repr += soft_weight[i] * encoded_layers[i]

        return wtd_encoded_repr

    def span_diff(self, encoded_input, start_idx, end_idx):
        """Does the difference based span representation: h_j - h_i
        encoded_input: B x L x H
        start_idx: integer
        end_idx: integer
        """
        span_repr = encoded_input[:, end_idx, :] - encoded_input[:, start_idx, :]
        return span_repr

    def span_avg(self, encoded_input, start_idx, end_idx):
        span_repr = 0
        span_length = (end_idx - start_idx + 1)
        assert(span_length > 0)
        for idx in range(start_idx, end_idx + 1):
            span_repr += encoded_input[:, idx, :]/span_length
        return span_repr


if __name__=='__main__':
    model = Encoder().cuda()
    tokenized_input = model.tokenize_input("Hello world!")  # 1 x L
    model.encode_tokens(tokenized_input).shape
