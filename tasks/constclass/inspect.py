import argparse
import os
import torch 
from tasks.constclass.main import *

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt-path', type=str, default='tasks/constclass/checkpoints/')
args = parser.parse_args()

for model_name in ['bert', 'roberta', 'spanbert', 'xlnet']:
    for model_size in ['base', 'large']:
        for encoding_method in ['attn', 'avg', 'coherent', 'coherent_original', 'diff', 'diff_sum', 'endpoint', 'max']:
            ckpt = torch.load(os.path.join(args.ckpt_path, f'{model_name}-{model_size}-cased-{encoding_method}.ckpt'))
            layer_weights = ckpt['weighing_params'].softmax(0).tolist()
            layer_weights_str = ' '.join(['{:.3f}'.format(item) for item in layer_weights])
            print(f'{model_name}-{model_size}-cased-{encoding_method}', layer_weights_str)

