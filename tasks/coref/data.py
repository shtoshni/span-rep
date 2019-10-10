from torchtext.data import Example, Field, Dataset
import torchtext.data as data
import json


class CorefDataset(Dataset):
    """Class for parsing the Ontonotes coref dataset."""

    def __init__(self, path, model, feedback=False,
                 encoding="utf-8", separator="\t",
                 max_seq_len=512):
        text_field = Field(sequential=True, use_vocab=False, include_lengths=True,
                           batch_first=True, pad_token=model.tokenizer.pad_token_id)
        non_seq_field = Field(sequential=False, use_vocab=False, batch_first=True)
        fields = [('text', text_field),
                  ('span1', non_seq_field),
                  ('span2', non_seq_field),
                  ('label', non_seq_field)]

        examples = []
        with open(path, encoding=encoding) as f:
            counter = 0
            for line in f:
                if feedback:
                    counter += 1
                    if counter > 1000:
                        break

                instance = json.loads(line)
                text, subword_to_word_idx = model.tokenize(
                    instance["text"].split(), get_subword_indices=True)

                for target in instance["targets"]:
                    span1_index = self.get_tokenized_span_indices(
                        subword_to_word_idx, target["span1"])
                    span2_index = self.get_tokenized_span_indices(
                        subword_to_word_idx, target["span2"])
                    label = target["label"]
                    examples.append(
                        Example.fromlist([text, span1_index, span2_index, label], fields))

        super(CorefDataset, self).__init__(examples, fields)

    @staticmethod
    def get_tokenized_span_indices(subword_to_word_idx, orig_span_indices):
        orig_start_idx, orig_end_idx = orig_span_indices
        start_idx = subword_to_word_idx.index(orig_start_idx)
        # Search for the index of the last subword
        end_idx = len(subword_to_word_idx) - 1 - subword_to_word_idx[::-1].index(orig_end_idx - 1)
        return [start_idx, end_idx]

    @staticmethod
    def sort_key(example):
        return len(example.text)

    @classmethod
    def iters(cls, path, model, batch_size=32, feedback=False):
        train, val, test = CorefDataset.splits(
            path=path, train='train.json', validation='development.json', test='test.json',
            model=model, feedback=feedback)

        train_iter, val_iter, test_iter = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size,
            sort_within_batch=True, shuffle=True, repeat=False)

        return (train_iter, val_iter, test_iter)


if __name__ == '__main__':
    from encoders.pretrained_transformers import Encoder
    encoder = Encoder(cased=False)
    path = "/share/data/lang/users/freda/codebase/hackathon_2019/tasks/constituent/data/edges/ontonotes/coref/"
    train_iter, val_iter, test_iter = CorefDataset.iters(path, encoder, feedback=True)

    for batch_data in train_iter:
        print(batch_data.text[0].shape)
        print(batch_data.span1.shape)
        print(batch_data.span2.shape)
        print(batch_data.label.shape)

        text, text_len = batch_data.text
        text_ids = text[0, :text_len[0]].tolist()

        itos = encoder.tokenizer.ids_to_tokens
        sent_tokens = [itos[text_id] for text_id in text_ids]
        sent = ' '.join(sent_tokens)
        span1 = ' '.join(sent_tokens[batch_data.span1[0, 0]:batch_data.span1[0, 1] + 1])
        print(sent)
        print(batch_data.span1[0, :])
        print(span1)
        break
