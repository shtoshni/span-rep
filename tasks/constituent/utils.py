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


def instance_f1_info(label, preds):
    label_ones = label.sum().item()
    preds_ones = preds.sum().item()
    correct_ones = (label * preds).sum().item()
    return correct_ones, preds_ones, label_ones


def f1_score(numerator, denom_p, denom_r):
    if numerator == 0:
        return 0
    p = float(numerator) / denom_p
    r = float(numerator) / denom_r
    return 2 * p * r / (p + r)
