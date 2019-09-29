import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
from os import path

from pathlib import Path
from encoders import Encoder
import pickle
import random
import re
import subprocess
import time

import numpy as np
import torch
from tqdm import tqdm
import argparse


from dataset import load_data, ID_TO_LABEL, MASK_LABEL


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Add arguments to parser
    parser.add_argument('-pretrained_model', default='bert',
                        help='Pretrained model', type=str)
    parser.add_argument('-model_type', default='base',
                        help='Specific model type', type=str)
    parser.add_argument('-fine_tune', default=False, action="store_true",
                        help='If true then finetune the pretrained model.')

    return parser.parse_args()


class PretrainedModelNER(torch.nn.Module):
    def __init__(self, model='bert', model_type='base', fine_tune=False):
        super().__init__()
        self.encoder = Encoder(model=model, model_type=model_type,
                               fine_tune=fine_tune)
        self.tokenizer = self.encoder.tokenizer

        self.linear = torch.nn.Linear(
            self.encoder.hidden_size, len(ID_TO_LABEL))

    def forward(self, batch_indices):
        model_output = self.encoder(
            batch_indices, just_last_layer=True)
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

def evaluate(out_dir, model, data):
    proc_in = []

    tokens = 0
    correct = 0
    for examples, masks, labels, label_masks in batcher(*data, batch_size=32):
        with torch.no_grad():
            logits = model(torch.tensor(examples).cuda()).cpu().detach().numpy()
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


def main(pretrained_model, model_type, fine_tune):
    num_epochs = 20

    model = PretrainedModelNER(
        model=pretrained_model, model_type=model_type,
        fine_tune=fine_tune)
    model.cuda()

    tokenizer = model.tokenizer

    data = load_data(Path('data'), tokenizer=tokenizer, bio=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss = torch.nn.CrossEntropyLoss(reduction='none')

    best_f1 = -1

    for epoch in range(num_epochs):
        start = time.time()
        epoch_loss = 0
        n_batches = 0

        model.train()
        for examples, masks, labels, label_masks in tqdm(list(batcher(*data['train'], shuffle=True, batch_size=32))[:200]):
            optimizer.zero_grad()
            classifier_log_odds = model(torch.tensor(examples).cuda())

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
        model.eval()
        f1 = evaluate(Path('output'), model, data['dev'])
        eval_time = time.time() - eval_start
        if f1 > best_f1:
            best_f1 = f1

        # Training loss is technically computed slightly incorrectly here but it's just there for sanity checking anyways
        logger.info(f'Epoch: {epoch} Dev F1: {f1:0.04f} Training Loss: {epoch_loss / n_batches:0.04f} Duration: {train_time:0.1f} sec Eval duration: {eval_time:0.1f} sec')

if __name__ == '__main__':
    args = get_args()
    main(args.pretrained_model, args.model_type, args.fine_tune)
