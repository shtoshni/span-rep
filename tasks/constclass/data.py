
import json
import torch
from torch.utils.data import Dataset

from tasks.constclass.utils import convert_word_to_subword


class ConstituentDataset(Dataset):
    label_dict = dict()
    encoder = None

    def __init__(self, path, encoder):
        super(ConstituentDataset, self).__init__()
        raw_data = [json.loads(line) for line in open(path)]
        self.set_encoder(encoder)
        # preprocess
        self.data = list()
        for sentence in raw_data:
            text = sentence['text']
            tokenized_input, subword2word = encoder.tokenize_sentence(
                text, get_subword_indices=True, force_split=True)
            start_ids, end_ids = convert_word_to_subword(
                subword2word,
                torch.tensor([
                    item['span1'] for item in sentence['targets']
                ]).long().view(-1, 2),
                encoder.start_shift
            )
            tokenized_span_ranges = torch.cat(
                (start_ids.view(-1, 1), end_ids.view(-1, 1)), dim=1)
            for i, item in enumerate(sentence['targets']):
                label = item['label']
                self.add_label(label)
                self.data.append(
                    {
                        'text_ids': tokenized_input.cpu(),
                        'span': tokenized_span_ranges[i].view(1, -1).cpu(),
                        'label': self.label_dict[label]
                    }
                )
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]['text_ids'], self.data[index]['span'], \
            self.data[index]['label']

    @classmethod
    def add_label(cls, label):
        if label not in cls.label_dict:
            cls.label_dict[label] = len(cls.label_dict)

    @classmethod
    def set_encoder(cls, encoder):
        cls.encoder = encoder
        

def collate_fn(data):
    sents, spans, labels = list(zip(*data))
    max_length = max(item.shape[1] for item in sents)
    pad_id = ConstituentDataset.encoder.tokenizer.pad_token_id
    batch_size = len(sents)
    padded_sents = pad_id * torch.ones(batch_size, max_length).long()
    for i, sent in enumerate(sents):
        padded_sents[i, :sent.shape[1]] = sent[0, :]
    spans = torch.cat(spans, dim=0)
    labels = torch.tensor(labels)
    return padded_sents, spans, labels


# unit test
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from encoders.pretrained_transformers import Encoder
    encoder = Encoder('bert', 'base', True)
    for split in ['train', 'development', 'test']:
        dataset = ConstituentDataset(
            f'tasks/constclass/data/debug/{split}.json', 
            encoder
        )
        data_loader = DataLoader(dataset, 64, collate_fn=collate_fn)
        for sents, spans, labels in data_loader:
            pass
        print(
            f'Split "{split}" has passed the unit test '
            f'with {len(dataset)} instances.'
        )
    from IPython import embed; embed(using=False)
