from collections import defaultdict
from functools import partial
from pathlib import Path
import random

DOCSTART = '-DOCSTART-'
MASK_LABEL = 'zzz'

LABEL_MAP = {}
LABEL_MAP['O'] = 0
for ent_type in ('ORG', 'PER', 'LOC', 'MISC'):
    for token_type in ('S', 'B', 'I', 'E'):
        LABEL_MAP[f'{token_type}-{ent_type}'] = len(LABEL_MAP)
LABEL_MAP[MASK_LABEL] = len(LABEL_MAP)

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

INVALID_ENT = 'invalid'

def bioes_to_bio(tag_sequence):
    last_type = None
    converted_seq = []
    for tag in tag_sequence:
        fields = tag.split('-')
        if len(fields) == 1:
            converted_tag = tag
            if tag == 'O':
                last_type = 'O'
        else:
            token_type, ent_type = fields
            if token_type == 'S' or token_type == 'B':
                token_type = 'I' if last_type == 'O' or last_type is None else 'B'
            elif token_type == 'I' or token_type == 'E':
                token_type = 'I'
            else:
                raise ValueError(f'Invalid tag: {tag}')

            converted_tag = f'{token_type}-{ent_type}'
            last_type = token_type
        converted_seq.append(converted_tag)
    return converted_seq

def bioes_ids_to_bio(id_sequence):
    return [LABEL_MAP[t] for t in bioes_to_bio(ID_TO_LABEL[id_] for id_ in id_sequence)]

def eval_f1(labels, predictions):
    correct = 0
    recall_total = 0
    precision_total = 0

    total_tags = 0

    for gold_sequence, pred_sequence in zip(labels, predictions):
        curr_gold_ent = None
        gold_ent_start = None

        curr_pred_ent = None
        pred_ent_start = None

        ex_gold_entities = defaultdict(set)
        ex_pred_entities = defaultdict(set)
        index = 0

        for correct_tag, pred_tag in zip(gold_sequence, pred_sequence):
            correct_token_type, *gold_ent_type = correct_tag.split('-')
            pred_token_type, *pred_ent_type = pred_tag.split('-')

            if curr_gold_ent is not None and (correct_token_type == 'O' or correct_token_type == 'B'):
                ex_gold_entities[curr_gold_ent].add((gold_ent_start, index))
                curr_gold_ent = None
                gold_ent_start = None

            if correct_token_type == 'B' or (correct_token_type == 'I' and curr_gold_ent is None):
                curr_gold_ent = gold_ent_type[0]
                gold_ent_start = index

            if curr_pred_ent is not None and (pred_token_type == 'O' or pred_token_type == 'B'):
                ex_pred_entities[curr_pred_ent].add((pred_ent_start, index))
                curr_pred_ent = None
                pred_ent_start = None

            if pred_token_type == 'B' or (pred_token_type == 'I' and curr_pred_ent is None):
                curr_pred_ent = pred_ent_type[0]
                pred_ent_start = index
            elif pred_token_type == 'I' and pred_ent_type[0] != curr_pred_ent:
                curr_pred_ent = INVALID_ENT

            index += 1

        if curr_gold_ent is not None:
            ex_gold_entities[curr_gold_ent].add((gold_ent_start, index))

        if curr_pred_ent is not None:
            ex_pred_entities[curr_pred_ent].add((pred_ent_start, index))

        """
        print('-----------------')
        print(ex_precision_correct, ex_precision_total, ex_recall_correct, ex_recall_total)
        print()
        print(' '.join(next(k for k, v in LABEL_MAP.items() if v == idx) for m, idx in zip(mask_ex, label_ex) if m))
        print()
        print(' '.join(next(k for k, v in LABEL_MAP.items() if v == idx) for m, idx in zip(mask_ex, pred_ex) if m))
        """

        ex_correct = sum(len(ex_gold_entities[ent_type] & ex_pred_entities[ent_type]) for ent_type in ex_pred_entities)


        ex_precision_total = sum(len(s) for s in ex_pred_entities.values())
        ex_recall_total = sum(len(s) for s in ex_gold_entities.values())

        correct += ex_correct
        precision_total += ex_precision_total
        recall_total += ex_recall_total

    if precision_total == 0 or recall_total == 0:
        return 0

    recall = correct / recall_total
    precision = correct / precision_total
    if precision == 0 or recall == 0:
        return 0

    return 2 * recall * precision / (recall + precision)

def eval_accuracy(labels, predictions, ent_type_only=False):
    correct = 0
    total = 0
    for gold_sequence, pred_sequence in zip(labels, predictions):
        for correct_tag, pred_tag in zip(gold_sequence, pred_sequence):
            correct += (
                (correct_tag.split('-')[1:] == pred_tag.split('-')[1:])
                if ent_type_only else
                (correct_tag == pred_tag)
            )
            total += 1

    return correct / total

def eval_perf(labels, predictions, mode='f1'):
    if mode == 'f1':
        return eval_f1(labels,predictions)
    elif mode == 'accuracy':
        return eval_accuracy(labels,predictions)
    elif mode == 'type_only':
        return eval_accuracy(labels,predictions, ent_type_only=True)
    else:
        raise ValueError(f'Unrecognized evaluation mode; {mode}')

def load_data(path : Path, truncate=None, bio=False, tokenizer=None):

    data = {}
    for fold in ('train', 'dev', 'test'):
        fold_path = path / f'eng.{fold}.bioes.conll'
        if not fold_path.exists():
            continue

        examples = []
        labels = []
        label_masks = []

        sent = ['[CLS]']
        sent_labels = []
        label_mask = []
        print(f'Loading fold: {fold}')
        with fold_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    if sent and len(sent) > 1:
                        sent.append('[SEP]')
                        examples.append(tokenizer.convert_tokens_to_ids(sent))
                        if bio:
                            sent_labels = bioes_ids_to_bio(sent_labels)
                        labels.append(sent_labels)
                        label_masks.append(label_mask)
                        if truncate and len(examples) == truncate:
                            break
                    sent = ['[CLS]']
                    sent_labels = []
                    label_mask = []
                    continue
                word, _, _, tag = line.split()
                if word == DOCSTART:
                    continue

                tokens = tokenizer.tokenize(word)
                sent.extend(tokens)
                # sent.append('[SEP]')
                label_mask.append(1)
                sent_labels.append(LABEL_MAP[tag])

                label_mask.extend(0 for _ in range(len(tokens) - 1))
                sent_labels.extend(LABEL_MAP[MASK_LABEL] for _ in range(len(tokens) - 1))
        data[fold] = (examples, labels, label_masks)

    return data


if __name__ == '__main__':
    data = load_data(Path('data'))
    examples, labels, label_masks = data['train']
    print(f'Train examples: {len(examples)} Train tokens: {sum(len(ex) for ex in examples)} Max train len: {max(len(ex) for ex in examples)} labels: {len(LABEL_MAP)}')
    print([*LABEL_MAP])
