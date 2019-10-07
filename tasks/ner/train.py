import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from model import Net
from data_load import NerDataset, pad, VOCAB, idx2tag
import os
import sys
from os import path
import numpy as np
import argparse
import subprocess
import re
import hashlib
from collections import OrderedDict

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def train(model, iterator, optimizer, optimizer_add, criterion, tokenizer, max_gradient_norm=1.0):
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch
        _y = y  # for monitoring
        optimizer.zero_grad()
        optimizer_add.zero_grad()
        logits, y, _ = model(x, y)  # logits: (N, T, VOCAB), y: (N, T)

        logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)

        loss = criterion(logits, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_gradient_norm)

        optimizer.step()
        optimizer_add.step()

        if i == 0:
            print("=====sanity check======")
            print("words:", words[0])
            print("x:", x.cpu().numpy()[0][:seqlens[0]])
            print("tokens:", tokenizer.convert_ids_to_tokens(x.tolist()[0])[:seqlens[0]])
            print("is_heads:", is_heads[0])
            print("y:", _y.cpu().numpy()[0][:seqlens[0]])
            print("tags:", tags[0])
            print("seqlen:", seqlens[0])
            print("=======================")

        if i % 50 == 0:  # monitoring
            logging.info(f"step: {i}, loss: {loss.item()}")
            sys.stdout.flush()


def eval(model, iterator, f, model_path):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch

            _, _, y_hat = model(x, y)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    # gets results and save
    temp1_file = path.join(model_path, "temp")
    temp2_file = path.join(model_path, "temp_2")
    with open(temp1_file, 'w') as fout, open(temp2_file, 'w') as fout_2:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds) == len(words.split()) == len(tags.split())
            start_idx = model.encoder.start_shift
            end_idx = model.encoder.end_shift
            for w, t, p in zip(words.split()[start_idx:-end_idx],
                               tags.split()[start_idx:-end_idx],
                               preds[start_idx:-end_idx]):
                fout.write(f"{w} {t} {p}\n")
                fout_2.write(f"{t} {p}\n")
            fout.write("\n")
            fout_2.write("\n")

    eval_proc = subprocess.Popen(['conlleval'], stdin=open(temp2_file),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = eval_proc.communicate()

    full_log = stdout.decode('ascii')
    line = full_log.split('\n')[1].strip()
    if not line.startswith('accuracy'):
        raise ValueError(f'conlleval script gave bad output\n{stdout}')

    f1 = float(re.search('\S+$', line).group(0))
    print(f'F1: {f1}')

    final = f + ".F%.3f_full_log" % (f1)
    final_2 = f + ".F%.3f" % (f1)
    with open(final, 'w') as fout, open(final_2, 'w') as fout_2:
        result = open(temp1_file, "r").read()
        fout.write(f"{result}\n")
        fout.write(f"{full_log}")

        result = open(temp2_file, "r").read()
        fout_2.write(f"{result}")

    os.remove(temp1_file)
    os.remove(temp2_file)
    return f1


def get_model_name(hp):
    opt_dict = OrderedDict()
    # Only include important options in hash computation
    imp_opts = ['model', 'model_size', 'batch_size',
                'n_epochs',  'finetuning', 'top_rnns',
                'seed', 'lr', 'lr_add']
    for key, val in vars(hp).items():
        if key in imp_opts:
            opt_dict[key] = val
            print("%s\t%s" % (key, val))

    str_repr = str(opt_dict.items())
    hash_idx = hashlib.md5(str_repr.encode("utf-8")).hexdigest()
    model_name = "ner_" + str(hash_idx)
    return model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-lr", type=float, default=1e-5)
    parser.add_argument("-lr_add", type=float, default=2e-4)
    parser.add_argument("-n_epochs", type=int, default=10)
    parser.add_argument("-model", type=str, default='bert')
    parser.add_argument("-model_size", type=str, default='base')
    parser.add_argument("-finetuning", dest="finetuning", action="store_true")
    parser.add_argument("-top_rnns", dest="top_rnns", action="store_true")
    parser.add_argument("-logdir", type=str, default="checkpoints")
    parser.add_argument("-datadir", type=str,
                        default="/home/shtoshni/Research/hackathon_2019/tasks/ner/conll2003")
    parser.add_argument("-seed", type=int, default=0, help="Random seed.")
    hp = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(hp.seed)
    np.random.seed(hp.seed)
    model = Net(model=hp.model, model_size=hp.model_size,
                top_rnns=hp.top_rnns, vocab_size=len(VOCAB),
                device=device, finetuning=hp.finetuning).cuda()
    model_name = get_model_name(hp)
    model_path = path.join(hp.logdir, model_name)
    best_model_path = path.join(model_path, 'best_models')
    if not path.exists(model_path):
        os.makedirs(model_path)
    if not path.exists(best_model_path):
        os.makedirs(best_model_path)

    config_file = path.join(model_path, "config.txt")
    with open(config_file, 'w') as f:
        for key, val in vars(hp).items():
            if "dir" not in key:
                f.write(str(key) + "\t" + str(val) + "\n")

    train_dataset = NerDataset(path.join(hp.datadir, "train.txt"), model.encoder)
    eval_dataset = NerDataset(path.join(hp.datadir, "valid.txt"), model.encoder)
    test_dataset = NerDataset(path.join(hp.datadir, "test.txt"), model.encoder)

    tokenizer = model.encoder.tokenizer

    train_iter = data.DataLoader(
        dataset=train_dataset, batch_size=hp.batch_size,
        shuffle=True, num_workers=4, collate_fn=pad)
    eval_iter = data.DataLoader(
        dataset=eval_dataset, batch_size=hp.batch_size,
        shuffle=False, num_workers=4, collate_fn=pad)
    test_iter = data.DataLoader(
        dataset=test_dataset, batch_size=hp.batch_size,
        shuffle=False, num_workers=4, collate_fn=pad)

    optimizer = optim.AdamW(model.encoder.parameters(), lr=hp.lr, weight_decay=0.0)
    optimizer_add = optim.AdamW(model.fc.parameters(), lr=hp.lr_add, weight_decay=0.0)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    max_f1 = 0
    for epoch in range(1, hp.n_epochs+1):
        if epoch == 1:
            print("\n%s\n" % model_path)
            model.print_model_info()
        train(model, train_iter, optimizer, optimizer_add, criterion, tokenizer)

        print(f"=========eval at epoch={epoch}=========")
        fname = os.path.join(model_path, "model")
        f1 = eval(model, eval_iter, fname, model_path)

        if max_f1 < f1:
            max_f1 = f1
            best_fname = os.path.join(best_model_path, "model")
            torch.save(model.state_dict(), f"{best_fname}.pt")
            print(f"weights were saved to {best_fname}.pt")

        torch.save(model.state_dict(), f"{fname}.pt")
        print(f"weights were saved to {fname}.pt")
        sys.stdout.flush()

    print(f"Max F1 {max_f1}")
    # Load the best model again
    print("Loading best model to evaluate on test data")
    model.load_state_dict(torch.load(f"{best_fname}.pt"))
    test_f1 = eval(model, test_iter, fname, model_path)
    print(f"Test F1 {test_f1}")

    summary_file = path.join(model_path, "final_report.txt")
    with open(summary_file, 'w') as f:
        f.write("Val F1: %.3f\n" % max_f1)
        f.write("Test F1: %.3f\n" % test_f1)
