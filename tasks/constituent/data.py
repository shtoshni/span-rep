
import json
import torch
from torch.utils.data import Dataset

from tasks.constituent.utils import convert_word_to_subword


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
            for item in sentence['targets']:
                # skip the 'TOP' label as it overlaps with other labels
                if item['label'] == 'TOP':
                    continue
                text = sentence['text']
                tokenized_input, subword2word = encoder.tokenize_sentence(
                    text, get_subword_indices=True, force_split=True)
                start_id, end_id = convert_word_to_subword(
                    subword2word, 
                    torch.tensor(item['span1']).long().view(1, -1),
                    encoder.start_shift
                )
                self.data.append(
                    {
                        'text_ids': tokenized_input,
                        'span': torch.cat((start_id, end_id), dim=0).view(1, -1),
                        'label': item['label']
                    }
                )
                self.add_label(item['label'])
                assert len(sentence['text'].split()) >= item['span1'][1]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]['text_ids'], self.data[index]['span'], \
            self.label_dict[self.data[index]['label']]

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
    if torch.cuda.is_available():
        sents = torch.cat([
            torch.cat(
                [
                    item, 
                    torch.tensor([pad_id] * (max_length - item.shape[1])).view(
                        1, -1).long().cuda()
                ], dim=1
            ) for item in sents
        ], dim=0)
    else:
        sents = torch.cat([
            torch.cat(
                [
                    item, 
                    torch.tensor([pad_id] * (max_length - item.shape[1])).view(
                        1, -1).long()
                ], dim=1
            ) for item in sents
        ], dim=0)
    spans = torch.cat(spans, dim=0)
    return sents, spans, labels


# unit test
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from encoders import Encoder
    encoder = Encoder('bert', 'base', True)
    for split in ['train', 'development', 'test']:
        dataset = ConstituentDataset(
            f'tasks/constituent/data/edges/ontonotes/const/'
            f'debug/{split}.json', 
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
