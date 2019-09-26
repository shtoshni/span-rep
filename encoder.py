import torch
import torch.nn as nn
import logging

from transformers import *

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

MODEL_LIST = ['bert', 'roberta', 'gpt2']
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

        # Set parameters required on top of pre-trained models
        self.weighing_params = nn.Parameter(torch.ones(self.num_layers))

        # Attention-based Span representation parameters - MIGHT NOT BE USED
        self.attention_weight = nn.Parameter(torch.ones(self.hidden_size))



for model in MODEL_LIST:
    if 'gpt2' in model:
        model_type_list = GPT2_MODEL_SIZES
    else:
        model_type_list = BERT_MODEL_SIZES
    for model_type in model_type_list:
        print (model_type)
        encoder = Encoder(model=model, model_type=model_type)
