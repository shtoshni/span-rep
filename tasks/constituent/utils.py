import torch


def convert_word_to_subword(subword2word, spans, start_shift):
    if torch.cuda.is_available():
        subword2word = subword2word.cuda()
        spans = spans.cuda()
    temp = torch.arange(subword2word.shape[1]).unsqueeze(0).expand_as(
        subword2word)
    start_ids = ((subword2word >= 0).long() *
        (subword2word < spans[:, 0].unsqueeze(1)).long()).sum(1)
    end_ids = ((subword2word >= 0).long() *
        (subword2word < spans[:, 1].unsqueeze(1)).long()).sum(1)
    return start_ids + start_shift, end_ids + start_shift