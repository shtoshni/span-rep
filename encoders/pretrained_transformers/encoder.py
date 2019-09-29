import os
from os import path
import sys

import torch
import torch.nn as nn
import logging

from transformers import BertModel, RobertaModel, GPT2Model
from transformers import BertTokenizer, RobertaTokenizer, GPT2Tokenizer

from SpanBERT import BertModel as SpanbertModel
from span_reprs import get_avg_repr, get_diff_repr, \
    get_max_pooling_repr, get_alternate_repr

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


MODEL_LIST = ['bert', 'spanbert', 'roberta', 'gpt2']
BERT_MODEL_SIZES = ['base', 'large']
GPT2_MODEL_SIZES = ['small', 'medium', 'large']


class Encoder(nn.Module):
    def __init__(self, model='bert', model_size='base', cased=True,
                 fine_tune=False):
        super(Encoder, self).__init__()
        assert(model in MODEL_LIST)

        self.base_name = model
        self.model = None
        self.tokenizer = None
        self.num_layers = None
        self.hidden_size = None

        # First initialize the model and tokenizer
        model_name = ''
        # Do we want the tokenizer to lower case or not
        do_lower_case = not cased
        # Model is one of the BERT variants
        if 'bert' in model:
            assert (model_size in BERT_MODEL_SIZES)
            model_name = model + "-" + model_size
            if model == 'bert' and not cased:
                # Only original BERT supports uncased models
                model_name += '-uncased'
            elif model == 'roberta':
                # RoBERTa model types have no casing suffix in HuggingFace map
                # So we don't modify the model name
                pass
            else:
                model_name += '-cased'

            if model == 'bert':
                self.model = BertModel.from_pretrained(
                    model_name, output_hidden_states=True)
                self.tokenizer = BertTokenizer.from_pretrained(
                    model_name, do_lower_case=do_lower_case)
            elif model == 'roberta':
                self.model = RobertaModel.from_pretrained(
                    model_name, output_hidden_states=True)
                self.tokenizer = RobertaTokenizer.from_pretrained(
                    model_name, do_lower_case=do_lower_case)
            elif model == 'spanbert':
                # Model is loaded in a different way
                # Earlier "pytorch_transformers" required a .tar.gz URL/file
                # Updated library "transformers" requires pytorch_model.bin
                # and config.json separately.
                self.model = SpanbertModel.from_pretrained(
                    model_name)
                # SpanBERT uses the same tokenizer as BERT
                self.tokenizer = BertTokenizer.from_pretrained(
                    model_name[4:], do_lower_case=do_lower_case)

            self.num_layers = self.model.config.num_hidden_layers
            self.hidden_size = self.model.config.hidden_size

        elif model == 'gpt2':
            assert (model_size in GPT2_MODEL_SIZES)
            model_name = model
            if model_size != "small":
                model_name += "-" + model_size

            self.model = GPT2Model.from_pretrained(
                model_name, output_hidden_states=True)
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                model_name, do_lower_case=do_lower_case)
            self.num_layers = self.model.config.n_layer
            self.hidden_size = self.model.config.n_embd

        # Set the model name
        self.model_name = model_name

        # Set requires_grad to False if not fine tuning
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False

        # Set parameters required on top of pre-trained models
        self.weighing_params = nn.Parameter(torch.ones(self.num_layers))

        # Attention-based Span representation parameters - MIGHT NOT BE USED
        self.attention_weight = nn.Parameter(torch.ones(self.hidden_size))

    def tokenize(self, sentence, get_subword_indices=False):
        tokenizer = self.tokenizer
        subword_to_word_idx = []

        if not get_subword_indices:
            # Operate directly on a string
            if type(sentence) is list:
                sentence = ' '.join(sentence)
            token_ids = tokenizer.encode(
                sentence, add_special_tokens=(
                    False if self.base_name == 'gpt2' else True)
                )
            return token_ids

        elif get_subword_indices and self.base_name in ['bert', 'spanbert']:
            # We check for model name since Roberta/GPT2 don't operate at
            # word level

            # Convert sentence to a list of words
            if type(sentence) is list:
                # If list then don't do anything
                pass
            else:
                # Perform basic tokenization to get list of words
                sentence = tokenizer.basic_tokenizer.tokenize(sentence)

            token_ids = []
            for word_idx, word in enumerate(sentence):
                subword_list = tokenizer.wordpiece_tokenizer.tokenize(word)
                subword_ids = tokenizer.convert_tokens_to_ids(subword_list)

                subword_to_word_idx += [word_idx] * len(subword_ids)
                token_ids += subword_ids

            final_token_ids = tokenizer.add_special_tokens_single_sequence(
                token_ids)

            # Add -1 to denote the special symbols
            subword_to_word_idx = ([-1] + subword_to_word_idx + [-1])
            return final_token_ids, subword_to_word_idx
        else:
            raise Exception("%s doesn't support getting word indices"
                            % self.base_name)

    def tokenize_sentence(self, sentence, get_subword_indices=False):
        output = self.tokenize(
            sentence, get_subword_indices=get_subword_indices)
        if get_subword_indices:
            return (torch.tensor(output[0]).unsqueeze(dim=0).cuda(),
                    torch.tensor(output[1]).unsqueeze(dim=0).cuda())
        else:
            return torch.tensor(output).unsqueeze(dim=0).cuda()

    def tokenize_batch(self, list_of_sentences, get_subword_indices=False):
        """
        sentence: a whole string containing all the tokens (NOT A LIST).
        """
        all_token_ids = []
        all_subword_to_word_idx = []

        sentence_len_list = []
        max_sentence_len = 0
        for sentence in list_of_sentences:
            if get_subword_indices:
                token_ids, subword_to_word_idx = \
                    self.tokenize(sentence, get_subword_indices=True)
                all_subword_to_word_idx.append(subword_to_word_idx)
            else:
                token_ids = self.tokenize(sentence)

            all_token_ids.append(token_ids)
            sentence_len_list.append(len(token_ids))
            if max_sentence_len < sentence_len_list[-1]:
                max_sentence_len = sentence_len_list[-1]

        # Pad the sentences to max length
        pad_token = (self.tokenizer.eos_token_id if self.base_name == 'gpt2'
                     else self.tokenizer.pad_token_id)
        all_token_ids = [
            (token_ids + (max_sentence_len - len(token_ids)) * [pad_token])
            for token_ids in all_token_ids
        ]

        if get_subword_indices:
            all_subword_to_word_idx = [
                (word_indices + (max_sentence_len - len(word_indices)) * [-1])
                for word_indices in all_subword_to_word_idx
            ]

        # Tensorize the list
        batch_token_ids = torch.tensor(all_token_ids)
        batch_lens = torch.tensor(sentence_len_list)
        batch_token_ids, batch_lens = batch_token_ids.cuda(), batch_lens.cuda()
        if get_subword_indices:
            return (batch_token_ids, batch_lens,
                    torch.tensor(all_subword_to_word_idx))
        else:
            return (batch_token_ids, batch_lens)

    def forward(self, batch_ids, just_last_layer=False):
        """
        Encode a batch of token IDs.
        batch_ids: B x L
        """
        pad_token = (self.tokenizer.eos_token_id if self.base_name == 'gpt2'
                     else self.tokenizer.pad_token_id)
        input_mask = (batch_ids != pad_token).cuda().float()
        if 'spanbert' in self.model_name:
            # SpanBERT is based on old APIs
            encoded_layers, _ = self.model(
                batch_ids, attention_mask=input_mask)
            last_layer_states = encoded_layers[-1]
        else:
            last_layer_states, _,  encoded_layers = self.model(
                batch_ids, attention_mask=input_mask)  # B x L x E
            # Encoded layers also has the embedding layer - 0th entry
            encoded_layers = encoded_layers[1:]

        if just_last_layer:
            return last_layer_states
        else:

            wtd_encoded_repr = 0
            soft_weight = nn.functional.softmax(self.weighing_params, dim=0)

            for i in range(self.num_layers):
                wtd_encoded_repr += soft_weight[i] * encoded_layers[i]

            return wtd_encoded_repr


if __name__ == '__main__':
    model = Encoder(model='gpt2', model_size='small').cuda()
    tokenized_input, input_lengths = model.tokenize_batch(
        ["Hello unforgiving world!", "What's up"], get_subword_indices=False)
    output = model(tokenized_input)

    for idx in range(tokenized_input.shape[0]):
        print(model.tokenizer.convert_ids_to_tokens(
            tokenized_input[idx, :].tolist()))
    print(get_avg_repr(output, 0, 2).shape)
    print(get_diff_repr(output, 1, 3).shape)
