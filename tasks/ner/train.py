import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from model import Net
from data_load import NerDataset, pad, VOCAB, tag2idx, idx2tag
import os
import numpy as np
import argparse
import subprocess
import re
from tqdm import tqdm


def train(model, iterator, optimizer, criterion, tokenizer):
    model.train()
    for i, batch in tqdm(enumerate(iterator)):
        words, x, is_heads, tags, y, seqlens = batch
        _y = y  # for monitoring
        optimizer.zero_grad()
        logits, y, _ = model(x, y)  # logits: (N, T, VOCAB), y: (N, T)

        logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)

        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

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
            print(f"step: {i}, loss: {loss.item()}")


def eval(model, iterator, f):
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
    with open("temp", 'w') as fout, open("temp_2", 'w') as fout_2:
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

    eval_proc = subprocess.Popen(['conlleval'], stdin=open("temp_2"),
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
        result = open("temp", "r").read()
        fout.write(f"{result}\n")
        fout.write(f"{full_log}")

        result = open("temp_2", "r").read()
        fout_2.write(f"{result}")

    os.remove("temp")
    os.remove("temp_2")
    return f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-lr", type=float, default=0.0001)
    parser.add_argument("-n_epochs", type=int, default=10)
    parser.add_argument("-model", type=str, default='bert')
    parser.add_argument("-model_size", type=str, default='base')
    parser.add_argument("-finetuning", dest="finetuning", action="store_true")
    parser.add_argument("-top_rnns", dest="top_rnns", action="store_true")
    parser.add_argument("-logdir", type=str, default="checkpoints/01")
    parser.add_argument("-trainset", type=str, default="conll2003/train.txt")
    parser.add_argument("-validset", type=str, default="conll2003/valid.txt")
    hp = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Net(model=hp.model, model_size=hp.model_size,
                top_rnns=hp.top_rnns, vocab_size=len(VOCAB),
                device=device, finetuning=hp.finetuning).cuda()

    train_dataset = NerDataset(hp.trainset, model.encoder)
    eval_dataset = NerDataset(hp.validset, model.encoder)

    tokenizer = model.encoder.tokenizer

    train_iter = data.DataLoader(
        dataset=train_dataset, batch_size=hp.batch_size,
        shuffle=True, num_workers=4, collate_fn=pad)
    eval_iter = data.DataLoader(
        dataset=eval_dataset, batch_size=hp.batch_size,
        shuffle=False, num_workers=4, collate_fn=pad)

    # optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    optimizer = optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    max_f1 = 0
    for epoch in range(1, hp.n_epochs+1):
        if epoch == 1:
            model.print_model_info()
        train(model, train_iter, optimizer, criterion, tokenizer)

        print(f"=========eval at epoch={epoch}=========")
        if not os.path.exists(hp.logdir):
            os.makedirs(hp.logdir)
        fname = os.path.join(hp.logdir, str(epoch))
        f1 = eval(model, eval_iter, fname)

        if max_f1 < f1:
            max_f1 = f1
        torch.save(model.state_dict(), f"{fname}.pt")
        print(f"weights were saved to {fname}.pt")

    print(f"Max F1 {max_f1}")
