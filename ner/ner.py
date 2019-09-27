import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
from os import path
sys.path.append(path.join(sys.path[0], '../'))

from pathlib import Path
import pickle
import random
import re
import subprocess
import time

import numpy as np
import torch
from tqdm import tqdm
import argparse

from encoder import Encoder

from dataset import load_data, ID_TO_LABEL, MASK_LABEL


def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Add arguments to parser
    parser.add_argument('-model', default='bert',
                        help='Pretrained model', type=str)
    parser.add_argument('-model_type', default='base',
                        help='Specific model type', type=str)


class BertNER(torch.nn.Module):
    def __init__(self, hidden_size=768, num_layers=12):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, len(ID_TO_LABEL))

    def forward(self, model_output):
        return self.linear(model_output)[:, 1:-1, :]

def batcher(examples, labels, label_masks, batch_size=32, shuffle=False):
    indices = list(range(len(examples)))
    if shuffle:
        random.shuffle(indices)

    for batch_start in range(0, len(examples) // batch_size * batch_size, batch_size):
        batch_indices = indices[batch_start:(batch_start + batch_size)]

        max_len = max(len(examples[index]) for index in batch_indices)
        example_arr = np.zeros((batch_size, max_len), dtype=np.int64)
        mask_arr = np.zeros((batch_size, max_len), dtype=np.int64)
        label_arr = np.zeros((batch_size, max_len - 2), dtype=np.int64)
        label_mask_arr = np.zeros((batch_size, max_len - 2), dtype=np.int64)
        for row, index in enumerate(batch_indices):
            ex = examples[index]
            # Shift length by two to account for CLS and SEP tokens
            ex_len = len(ex) - 2

            example_arr[row, :ex_len + 2] = ex
            mask_arr[row, :ex_len + 2] = 1
            label_arr[row, :ex_len] = labels[index]
            label_mask_arr[row, :ex_len] = label_masks[index]

        yield example_arr, mask_arr, label_arr, label_mask_arr

def evaluate(out_dir, bert_model, classifier, data):
    proc_in = []

    tokens = 0
    correct = 0
    for examples, masks, labels, label_masks in batcher(*data, batch_size=128):
        with torch.no_grad():
            bert_output = bert_model.encode_tokens(torch.tensor(examples).cuda())

            logits = classifier(bert_output).cpu().detach().numpy()
            predictions = np.argmax(logits, axis=-1)
            correct += np.sum((predictions == labels) * label_masks)
            tokens += np.sum(label_masks)

            for gold_seq, pred_seq in zip(predictions, labels):
                for correct_id, pred_id in zip(gold_seq, pred_seq):
                    correct_tag = ID_TO_LABEL[correct_id]
                    pred_tag = ID_TO_LABEL[pred_id]
                    if correct_tag == MASK_LABEL:
                        continue
                    if pred_tag == MASK_LABEL:
                        pred_tag = 'O'
                    proc_in.append(f'{correct_tag} {pred_tag}\n')
                proc_in.append('\n')


    eval_proc = subprocess.Popen(['conll/bin/conlleval'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in proc_in:
        eval_proc.stdin.write(line.encode('ascii'))
    stdout, stderr = eval_proc.communicate()
    eval_proc.stdin.close()

    line = stdout.decode('ascii').split('\n')[1].strip()
    if not line.startswith('accuracy'):
        raise ValueError(f'conlleval script gave bad output\n{stdout}')

    f1 = float(re.search('\S+$', line).group(0))
    print(f'F1: {f1}')

    return f1


def main():
    num_epochs = 20

    encoder = Encoder(fine_tune=True)
    bert_model = encoder.cuda()

    classifier = BertNER(hidden_size=encoder.hidden_size,
                         num_layers=encoder.num_layers)
    classifier.cuda()

    tokenizer = encoder.tokenizer

    data = load_data(Path('data'), tokenizer=tokenizer, bio=True)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)

    loss = torch.nn.CrossEntropyLoss(reduction='none')

    best_f1 = -1

    for epoch in range(num_epochs):
        start = time.time()
        epoch_loss = 0
        n_batches = 0

        for examples, masks, labels, label_masks in tqdm(list(batcher(*data['train'], shuffle=True, batch_size=16))):
            optimizer.zero_grad()
            bert_output = encoder.encode_tokens(torch.tensor(examples).cuda())
            classifier_log_odds = classifier(bert_output)

            reshaped_output = classifier_log_odds.reshape((-1, classifier_log_odds.shape[-1]))
            reshaped_labels = torch.tensor(labels).cuda().reshape((-1,))
            reshaped_label_mask = torch.tensor(label_masks).cuda().reshape((-1,)).float()


            full_loss = loss(reshaped_output, reshaped_labels)
            masked_loss = full_loss * reshaped_label_mask
            average_loss = masked_loss.sum() / reshaped_label_mask.sum()

            average_loss.backward()
            optimizer.step()

            n_batches += 1
            epoch_loss += average_loss.cpu().detach().numpy()
        train_time = time.time() - start

        eval_start = time.time()
        f1 = evaluate(Path('output'), bert_model, classifier, data['dev'])
        eval_time = time.time() - eval_start
        if f1 > best_f1:
            best_f1 = f1

        # Training loss is technically computed slightly incorrectly here but it's just there for sanity checking anyways
        logger.info(f'Epoch: {epoch} Dev F1: {f1:0.04f} Training Loss: {epoch_loss / n_batches:0.04f} Duration: {train_time:0.1f} sec Eval duration: {eval_time:0.1f} sec')

if __name__ == '__main__':
    main()
