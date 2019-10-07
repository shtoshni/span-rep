import argparse
import json
import os
import types

import torch

from diora.scripts.train import argument_parser, parse_args, configure
from diora.scripts.train import get_validation_dataset, get_validation_iterator
from diora.scripts.train import build_net

from diora.logging.configuration import get_logger

from diora.analysis.cky import ParsePredictor as CKY


def override_init_with_batch(var):
    init_with_batch = var.init_with_batch

    def func(self, *args, **kwargs):
        init_with_batch(*args, **kwargs)
        self.saved_scalars = {i: {} for i in range(self.length)}
        self.saved_scalars_out = {i: {} for i in range(self.length)}

    var.init_with_batch = types.MethodType(func, var)


def override_inside_hook(var):
    def func(self, level, h, c, s):
        length = self.length
        B = self.batch_size
        L = length - level

        assert s.shape[0] == B
        assert s.shape[1] == L
        # assert s.shape[2] == N
        assert s.shape[3] == 1
        assert len(s.shape) == 4
        smax = s.max(2, keepdim=True)[0]
        s = s - smax

        for pos in range(L):
            self.saved_scalars[level][pos] = s[:, pos, :]

    var.inside_hook = types.MethodType(func, var)


def replace_leaves(tree, leaves):
    def func(tr, pos=0):
        if not isinstance(tr, (list, tuple)):
            return 1, leaves[pos]

        newtree = []
        sofar = 0
        for node in tr:
            size, newnode = func(node, pos+sofar)
            sofar += size
            newtree += [newnode]

        return sofar, newtree

    _, newtree = func(tree)

    return newtree


def run(options):
    logger = get_logger()

    validation_dataset = get_validation_dataset(options)
    validation_iterator = get_validation_iterator(options, validation_dataset)
    word2idx = validation_dataset['word2idx']
    embeddings = validation_dataset['embeddings']

    idx2word = {v: k for k, v in word2idx.items()}

    logger.info('Initializing model.')
    trainer = build_net(options, embeddings, validation_iterator)

    # Parse

    diora = trainer.net.diora

    ## Monkey patch parsing specific methods.
    override_init_with_batch(diora)
    override_inside_hook(diora)

    ## Turn off outside pass.
    trainer.net.diora.outside = False

    ## Eval mode.
    trainer.net.eval()
    
    return trainer, word2idx
            


class DioraRepresentation(object):
    def __init__(self, corpus_path):
        self.corpus = corpus_path
        lines = [str(i) + ' ' + line for i, line in enumerate(open(self.corpus))]
        options = torch.load(os.path.join(os.path.dirname(__file__), 'options.pt'))
        options.elmo_cache_dir = os.path.join(os.path.dirname(__file__), options.elmo_cache_dir)
        print(options)
        with open(options.validation_path, 'w') as fout:
            for line in lines:
                fout.write(line)
            fout.close()
        self.trainer, self.word2idx = run(options)
        os.system('rm {:s}'.format(options.validation_path))


    # get fixed-length span representation repr(sent[start:end])
    def span_representation(self, sent, start, end):
        sent_tensor = torch.tensor([self.word2idx[item] for item in sent]).long().view(1, -1)
        sent_length = sent_tensor.shape[1]
        batch_map = {
            'sentences': sent_tensor,
            'neg_samples': sent_tensor, # just ignore this
            'batch_size': 1,
            'length': sent_length,
            'example_ids': ['0']
        }
        _ = self.trainer.step(batch_map, train=False, compute_loss=False)
        span_length = end - start
        chart = self.trainer.net.diora.chart.inside_h[0]
        span_index = sent_length * (span_length - 1) - (span_length - 1) * (span_length - 2) // 2 +  start 
        return chart[span_index]



if __name__ == '__main__':
    sent = 'hello world !'.split()
    tool = DioraRepresentation('./corpus.txt')
    representation = tool.span_representation(sent, 0, 3)
    print(representation.shape)

