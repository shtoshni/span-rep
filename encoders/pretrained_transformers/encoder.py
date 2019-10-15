import torch
import torch.nn as nn
import logging

from transformers import BertModel, RobertaModel, GPT2Model, XLNetModel
from transformers import BertTokenizer, RobertaTokenizer, GPT2Tokenizer, XLNetTokenizer

from encoders.pretrained_transformers.SpanBERT import BertModel as SpanbertModel
from encoders.pretrained_transformers.utils import get_sequence_mask

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

# Constants
MODEL_LIST = ['bert', 'spanbert', 'roberta', 'xlnet', 'gpt2']
BERT_MODEL_SIZES = ['base', 'large']
GPT2_MODEL_SIZES = ['small', 'medium', 'large']


class Encoder(nn.Module):
    def __init__(self, model='bert', model_size='base', cased=True,
                 fine_tune=False, use_proj=False, proj_dim=256):
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
                # Earlier "pytorch_transformers" required a .tar.gz URL/file.
                # Updated library "transformers" requires pytorch_model.bin and config.json
                # separately. That's why we have to keep the SpanBERT codebase around and initialize
                # the model using that codebase (based on pytorch_pretrained_bert).
                # NOTE: By default transformer models are initialized to eval() mode!
                # Not using the eval() mode will result in randomness.
                self.model = SpanbertModel.from_pretrained(
                    model_name).eval()
                # SpanBERT uses the same tokenizer as BERT (that's why the slicing in model name).
                # We use the tokenizer from "transformers" since it provides an almost unified API.
                self.tokenizer = BertTokenizer.from_pretrained(
                    model_name[4:], do_lower_case=do_lower_case)

            self.num_layers = self.model.config.num_hidden_layers
            self.hidden_size = self.model.config.hidden_size

        elif model == "xlnet":
            model_name = model + "-" + model_size + "-cased"
            self.model = XLNetModel.from_pretrained(model_name, output_hidden_states=True)
            self.tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
            self.num_layers = self.model.config.num_hidden_layers
            self.hidden_size = self.model.config.hidden_size
        elif model == 'gpt2':
            assert (model_size in GPT2_MODEL_SIZES)
            model_name = model
            if model_size != "small":
                model_name += "-" + model_size

            self.model = GPT2Model.from_pretrained(
                model_name, output_hidden_states=True)
            # Set the EOS token to be the PAD token since no explicit pad token
            # in GPT2 implementation.
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                model_name, do_lower_case=do_lower_case, pad_token="<|endoftext|>")

            self.num_layers = self.model.config.n_layer
            self.hidden_size = self.model.config.n_embd

        # Set the model name
        self.model_name = model_name

        # Set shift size due to introduction of special tokens
        if self.base_name == 'xlnet':
            self.start_shift = 0
            self.end_shift = 2
        else:
            self.start_shift = (1 if self.tokenizer._cls_token else 0)
            self.end_shift = (1 if self.tokenizer._sep_token else 0)

        # Set requires_grad to False if not fine tuning
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False

        if use_proj:
            # Apply a projection layer to output of pretrained models
            self.proj = nn.Linear(self.hidden_size, proj_dim)
            # Update the hidden size
            self.hidden_size = proj_dim
        else:
            self.proj = None
        # Set parameters required on top of pre-trained models
        self.weighing_params = nn.Parameter(torch.ones(self.num_layers))

        # Attention-based Span representation parameters - MIGHT NOT BE USED
        self.attention_params = nn.Linear(self.hidden_size, 1)
        nn.init.constant_(self.attention_params.weight, 0)

    def tokenize(self, sentence, get_subword_indices=False, force_split=False):
        """
        sentence: A single sentence where the sentence is either a string or list.
        get_subword_indices: Boolean indicating whether subword indices corresponding to words
            are needed as an output of tokenization or not. Useful for tagging tasks.
        force_split: When True splits the string using python's inbuilt split method; otherwise,
            uses more sophisticated tokenizers if possible.

        Returns: A list of length L, # of tokens, or a pair of L-length lists
            if get_subword_indices is set to True.
        """
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

        elif get_subword_indices:
            # Convert sentence to a list of words
            if type(sentence) is list:
                # If list then don't do anything
                pass
            elif force_split:
                sentence = sentence.strip().split()
            else:
                try:
                    sentence = tokenizer.basic_tokenizer.tokenize(sentence)
                except AttributeError:
                    # Basic tokenizer is not a part of Roberta and GPT2
                    sentence = sentence.strip().split()

            if self.base_name in ['bert', 'spanbert', 'xlnet']:
                token_ids = []
                for word_idx, word in enumerate(sentence):
                    subword_list = tokenizer.tokenize(word)
                    subword_ids = tokenizer.convert_tokens_to_ids(subword_list)

                    subword_to_word_idx += [word_idx] * len(subword_ids)
                    token_ids += subword_ids

            elif self.base_name in ['roberta', 'gpt2']:
                token_ids = []
                for word_idx, word in enumerate(sentence):
                    subword_list = tokenizer.tokenize(word, add_prefix_space=True)
                    subword_ids = tokenizer.convert_tokens_to_ids(subword_list)

                    subword_to_word_idx += [word_idx] * len(subword_ids)
                    token_ids += subword_ids
            else:
                raise Exception("%s doesn't support getting word indices"
                                % self.base_name)

            # In case the model is supported
            if self.base_name == 'gpt2':
                final_token_ids = token_ids
            else:
                final_token_ids = tokenizer.add_special_tokens_single_sequence(
                    token_ids)
            # Add -1 to denote the special symbols
            subword_to_word_idx = (
                [-1] * self.start_shift + subword_to_word_idx + [-1] * self.end_shift)
            return final_token_ids, subword_to_word_idx

    def tokenize_sentence(self, sentence, get_subword_indices=False, force_split=False):
        """
        sentence: A single sentence where the sentence is either a string or list.
        get_subword_indices: Boolean indicating whether subword indices corresponding to words
            are needed as an output of tokenization or not. Useful for tagging tasks.
        force_split: When True splits the string using python's inbuilt split method; otherwise,
            uses more sophisticated tokenizers if possible.

        Returns: A tensor of size (1 x L) or a pair of (1 x L) tensors if get_subword_indices.
        """
        output = self.tokenize(
            sentence, get_subword_indices=get_subword_indices,
            force_split=force_split
        )
        if get_subword_indices:
            return (torch.tensor(output[0]).unsqueeze(dim=0).cuda(),
                    torch.tensor(output[1]).unsqueeze(dim=0).cuda())
        else:
            return torch.tensor(output).unsqueeze(dim=0).cuda()

    def tokenize_batch(self, list_of_sentences, get_subword_indices=False, force_split=False):
        """
        list_of_sentences: List of sentences where each sentence is either a string or list.
        get_subword_indices: Boolean indicating whether subword indices corresponding to words
            are needed as an output of tokenization or not. Useful for tagging tasks.
        force_split: When True splits the string using python's inbuilt split method; otherwise,
            uses more sophisticated tokenizers if possible.

        Returns: Padded tensors of size (B x L) or a pair of (B x L) tensors if get_subword_indices.
        """
        all_token_ids = []
        all_subword_to_word_idx = []

        sentence_len_list = []
        max_sentence_len = 0
        for sentence in list_of_sentences:
            if get_subword_indices:
                token_ids, subword_to_word_idx = \
                    self.tokenize(
                        sentence, get_subword_indices=True,
                        force_split=force_split
                    )
                all_subword_to_word_idx.append(subword_to_word_idx)
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

        if get_subword_indices:
            all_subword_to_word_idx = [
                (word_indices + (max_sentence_len - len(word_indices)) * [-1])
                for word_indices in all_subword_to_word_idx
            ]

        # Tensorize the list
        batch_token_ids = torch.tensor(all_token_ids).cuda()
        batch_lens = torch.tensor(sentence_len_list).cuda()
        if get_subword_indices:
            return (batch_token_ids, batch_lens,
                    torch.tensor(all_subword_to_word_idx))
        else:
            return (batch_token_ids, batch_lens)

    def get_sentence_repr(self, encoded_input, sentence_lens, method='avg'):
        """Get the sentence encoding of a batch of hidden states.
        encoded_input: B x L_max x H: Output of the pretrained model
        sentence_lens: B: Length of sentences in the batch
        repr_method: Method to reduce the sentence representation to a single vector
        """
        # First get input mask
        batch_size, max_len, h_size = encoded_input.shape
        # Remove the [SEP] or </s> token from calculation for sentence length
        # This allows us to mask out the suffix entirely including padding symbols
        actual_sentence_lens = sentence_lens - self.end_shift
        encoded_input = encoded_input[:, :(max_len - self.end_shift), :]
        input_mask = get_sequence_mask(actual_sentence_lens).cuda().float()  # B x L
        input_mask = torch.unsqueeze(input_mask, dim=2).expand_as(encoded_input)
        # Mask/Zero out values beyond sentence length
        encoded_input = encoded_input * input_mask

        # Now remove the start token from calculation as well
        actual_sentence_lens = actual_sentence_lens - self.start_shift
        assert (torch.min(actual_sentence_lens) > 0)

        if method == 'avg':
            # Summing over the padded symbols won't change the actual sum since they have
            # been masked out
            sent_repr = torch.sum(encoded_input[:, self.start_shift:, :], dim=1)
            # Now divide by sentence lengths
            sent_repr = sent_repr/torch.unsqueeze(actual_sentence_lens, dim=1).float()
            return sent_repr
        elif method == 'max':
            # To avoid errors in max, we can use the mask to make the padded entries affect
            # max operation. We will just add the min entry to earlier zeroed out padded symbols.
            min_val = torch.min(encoded_input)
            encoded_input = encoded_input + (1 - input_mask) * min_val
            return torch.max(encoded_input[:, self.start_shift:, :], dim=1)[0]
        elif method == "attn":
            attn_mask = (1 - input_mask) * (-1e10)
            attn_logits = (self.attention_params(encoded_input[:, self.start_shift:, :])
                           + attn_mask[:, self.start_shift:, :])
            attention_wts = nn.functional.softmax(attn_logits, dim=1)
            return torch.sum(attention_wts * encoded_input[:, self.start_shift:, :], dim=1)
        else:
            # First get the end point hidden vectors
            h_start = encoded_input[:, self.start_shift, :]
            h_end = encoded_input[torch.arange(batch_size), sentence_lens - 1 - self.end_shift, :]

            if method == 'diff':
                return (h_end - h_start)
            elif method == 'diff_sum':
                # Used by Ouchi et al
                return torch.cat([h_end - h_start, h_end + h_start], dim=1)
            elif method == 'coherent':
                # Used by Seo et al - https://arxiv.org/pdf/1906.05807.pdf
                # Use a partition size of one fourth
                p_size = int(h_size/4)
                coherence_term = torch.sum(
                    h_start[:, 2*p_size:3*p_size] * h_end[:, 3*p_size:], dim=1, keepdim=True)
                return torch.cat(
                    [h_start[:, :p_size], h_end[:, p_size:2*p_size], coherence_term], dim=1)
            elif method == 'coref':
                attn_mask = (1 - input_mask) * (-1e10)
                attn_logits = (self.attention_params(encoded_input[:, self.start_shift:, :])
                               + attn_mask[:, self.start_shift:, :])
                attention_wts = nn.functional.softmax(attn_logits, dim=1)
                attention_term = torch.sum(attention_wts * encoded_input[:, self.start_shift:, :],
                                           dim=1)
                return torch.cat([h_start, h_end, attention_term], dim=1)

            return torch.cat([h_start, h_end], dim=1)

    def forward(self, batch_ids, just_last_layer=False):
        """
        Encode a batch of token IDs.
        batch_ids: B x L
        just_last_layer: If True return the last layer else return a (learned) wtd avg of layers.
        """
        input_mask = (batch_ids != self.tokenizer.pad_token_id).cuda().float()
        if 'spanbert' in self.model_name:
            # SpanBERT is based on old APIs
            encoded_layers = self.model(
                batch_ids, attention_mask=input_mask)
            last_layer_states = encoded_layers[-1]
        else:
            last_layer_states, _,  encoded_layers = self.model(
                batch_ids, attention_mask=input_mask)  # B x L x E
            # Encoded layers also has the embedding layer - 0th entry
            encoded_layers = encoded_layers[1:]

        if just_last_layer:
            output = last_layer_states
        else:

            wtd_encoded_repr = 0
            soft_weight = nn.functional.softmax(self.weighing_params, dim=0)

            for i in range(self.num_layers):
                wtd_encoded_repr += soft_weight[i] * encoded_layers[i]

            output = wtd_encoded_repr

        if self.proj:
            return self.proj(output)
        else:
            return output


if __name__ == '__main__':
    model = Encoder(model='spanbert', model_size='base', use_proj=False).cuda()
    tokenized_input, input_lengths = model.tokenize_batch(
        ["Hello beautiful world!", "Chomsky says hello."], get_subword_indices=False)
    output = model(tokenized_input)

    for idx in range(tokenized_input.shape[0]):
        print(model.tokenizer.convert_ids_to_tokens(
            tokenized_input[idx, :].tolist()))

    for method in ["avg", "max", "diff", "diff_sum", "coherent", "attn"]:
        print(method, model.get_sentence_repr(output, input_lengths, method=method).shape)

    # Sanity check since the attention weights are initialized to 0, the two reprs should match
    avg_repr = model.get_sentence_repr(output, input_lengths, method="avg")
    attn_repr = model.get_sentence_repr(output, input_lengths, method="attn")
    print("Diff: %.3f" % (torch.norm(avg_repr - attn_repr)))
