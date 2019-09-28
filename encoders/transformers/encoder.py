import os
from os import path

import torch
import torch.nn as nn
import logging
import SpanBERT

from transformers import BertModel, RobertaModel, GPT2Model
from transformers import BertTokenizer, RobertaTokenizer, GPT2Tokenizer


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


MODEL_LIST = ['bert', 'spanbert', 'roberta', 'gpt2']
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
        # Do we want the tokenizer to lower case or not
        do_lower_case = not cased
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
                self.model = SpanBERT.BertModel.from_pretrained(
                    model_name)
                # SpanBERT uses the same tokenizer as BERT
                self.tokenizer = BertTokenizer.from_pretrained(
                    model_name[4:], do_lower_case=do_lower_case)

            self.num_layers = self.model.config.num_hidden_layers
            self.hidden_size = self.model.config.hidden_size

        elif model == 'gpt2':
            assert (model_type in GPT2_MODEL_SIZES)
            model_name = model
            if model_type:
                model_name += "-" + model_type

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

    def tokenize(self, sentence, max_length=512):
        tokenizer = self.tokenizer
        if 'bert' in self.model_name:
            # Check if the sentence is a list of words or a string
            if type(sentence) is list:
                # If list then don't do anything
                pass
            else:
                # Perform basic tokenization to get list of
                sentence = tokenizer.basic_tokenizer.tokenize(sentence)

            assert (len(sentence) > 0)
            # Now the sentence is a list of words
            first_subword_idx_list = []
            token_ids = []
            for word_idx, word in enumerate(sentence):
                subword_list = tokenizer.wordpiece_tokenizer.tokenize(word)
                subword_ids = [
                    tokenizer._convert_token_to_id(subword)
                    for subword in subword_list
                ]

                first_subword_idx_list.append(len(token_ids))
                token_ids += subword_ids

            final_token_ids = tokenizer.add_special_tokens_single_sequence(
                token_ids)

            # Search for the token id corresponding to first subword
            # token in original text
            token_shift = final_token_ids.index(token_ids[0])
            first_subword_idx_list = [
                (token_shift + first_subword_idx)
                for first_subword_idx in first_subword_idx_list
            ]
            return final_token_ids, first_subword_idx_list
        else:
            assert(isinstance(sentence, str))
            return tokenizer.encode(sentence, add_special_tokens=True)

    def tokenize_batch(self, list_of_sentences):
        """
        sentence: a whole string containing all the tokens (NOT A LIST).
        """
        all_token_ids = []
        all_first_subword_idx_list = []

        sentence_len_list = []
        max_sentence_len = 0
        for sentence in list_of_sentences:
            if 'bert' in self.model_name:
                token_ids, first_subword_idx_list = \
                    self.tokenize(sentence)
                all_first_subword_idx_list.append(first_subword_idx_list)
            else:
                token_ids = self.tokenize(sentence)

            all_token_ids.append(token_ids)
            sentence_len_list.append(len(token_ids))
            if max_sentence_len < sentence_len_list[-1]:
                max_sentence_len = sentence_len_list[-1]

        # Pad the sentences to max length
        all_token_ids = [
            (token_ids + (max_sentence_len - len(token_ids)) * [self.tokenizer.pad_token_id])
            for token_ids in all_token_ids
        ]

        # Tensorize the list
        batch_token_ids = torch.tensor(all_token_ids).cuda()
        return batch_token_ids, all_first_subword_idx_list

    def forward(self, batch_ids, just_last_layer=False):
        """
        Encode a batch of token IDs.
        batch_ids: B x L
        """
        input_mask = (batch_ids != self.tokenizer.pad_token_id).cuda().float()
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

    def span_diff(self, encoded_input, start_idx, end_idx):
        """Does the difference based span representation: h_j - h_i
        encoded_input: B x L x H
        start_idx: integer
        end_idx: integer
        """
        span_repr = (encoded_input[:, end_idx, :]
                     - encoded_input[:, start_idx, :])
        return span_repr

    def span_avg(self, encoded_input, start_idx, end_idx):
        span_repr = 0
        span_length = (end_idx - start_idx + 1)
        assert(span_length > 0)
        for idx in range(start_idx, end_idx + 1):
            span_repr += encoded_input[:, idx, :]/span_length
        return span_repr


if __name__ == '__main__':
    model = Encoder(model='spanbert', model_type='base').cuda()
    tokenized_input, useful_list = model.tokenize_batch(
        ["Hello unforgiving world!", "What's up"])  # 1 x L
    print(tokenized_input)
    print(tokenized_input.shape)
    print(model(tokenized_input).shape)
    print(useful_list)
    for idx in range(tokenized_input.shape[0]):
        print(model.tokenizer.convert_ids_to_tokens(
            tokenized_input[idx, :].tolist()))
