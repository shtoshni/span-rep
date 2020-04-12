import argparse
import json
import logging
import numpy as np
import os
from scipy.stats import spearmanr
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from encoders.pretrained_transformers import Encoder
from tasks.constituent.data import ConstituentDataset, collate_fn
from tasks.constituent.models import SpanClassifier
from tasks.constituent.utils import instance_f1_info, f1_score
from tasks.constituent.main import LearningRateController


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
<script>
window.onload=function() LLLLB
var words = {:s};

$('#text').html($.map(words, function(w) LLLLB
  return '<span style="background-color:hsl(360,100%,' + (w.attention * 50 + 50) + '%)">' + w.word + ' </span>'
RRRRB))
RRRRB
</script>
</head>
<body>

<div id="text">text goes here</div>

</body>
</html>
"""

def forward_batch(model, data, valid=False):
    sents, spans, labels = data
    output = encoder(sents)
    preds = model(output, spans[:, 0], spans[:, 1] - 1)
    if valid:
        return preds
    else:
        loss = nn.BCELoss()(preds, labels.float())
        return preds, loss


if __name__ == '__main__':
    # arguments from snippets
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, 
        default='tasks/constituent/data/edges/ontonotes/const/nonterminal')
    parser.add_argument('--model-path', type=str, 
        default='tasks/constituent/checkpoints')
    parser.add_argument('--model-name', type=str, default='debug')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--log-step', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1111)
    # slurm supportive snippets 
    parser.add_argument('--time-limit', type=float, default=13800)
    # customized arguments
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256])
    parser.add_argument('--model-type', type=str, default='bert')
    parser.add_argument('--model-size', type=str, default='base')
    parser.add_argument('--uncased', action='store_false', dest='cased')
    parser.add_argument('--encoding-method', type=str, default='avg')
    parser.add_argument('--use-proj', action='store_true', default=False)
    parser.add_argument('--proj-dim', type=int, default=256)
    # output viz info
    parser.add_argument('--viz-path', type=str, default='tasks/constituent/viz/')
    args = parser.parse_args()
    
    # save arguments
    args.start_time = time.time()
    args_save_path = os.path.join(
        args.model_path, args.model_name + '.args.pt'
    )
    torch.save(args, args_save_path)

    # initialize random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(
        os.path.join(
            args.model_path, args.model_name + '.log'
        ), 
        'a'
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False
    
    # create data sets, tokenizers, and data loaders
    encoder = Encoder(args.model_type, args.model_size, 
        args.cased, use_proj=args.use_proj, proj_dim=args.proj_dim
    )
    data_loader_path = os.path.join(
        args.model_path, args.model_name + '.loader.pt'
    )
    
    if os.path.exists(data_loader_path):
        logger.info('Loading datasets.')
        data_info = torch.load(data_loader_path)
        data_loader = data_info['data_loader']
        ConstituentDataset.label_dict = data_info['label_dict']
        ConstituentDataset.encoder = encoder
    else:
        logger.info('Creating datasets.')
        data_set = dict()
        data_loader = dict()
        for split in ['train', 'development', 'test']:
            data_set[split] = ConstituentDataset(
                os.path.join(args.data_path, f'{split}.json'),
                encoder=encoder
            )
            data_loader[split] = DataLoader(data_set[split], args.batch_size, 
                collate_fn=collate_fn, shuffle=(split=='train'))
        torch.save(
            {
                'data_loader': data_loader,
                'label_dict': ConstituentDataset.label_dict
            },
            data_loader_path
        )

    # initialize models: MLP
    logger.info('Initializing models.')
    model = SpanClassifier(
        encoder, args.use_proj, args.proj_dim, args.hidden_dims, 
        len(ConstituentDataset.label_dict),
        pooling_method=args.encoding_method
    )
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        model = model.cuda()
    
    # initialize optimizer
    logger.info('Initializing optimizer.')
    params = [encoder.weighing_params] + list(model.parameters())
    optimizer = getattr(torch.optim, args.optimizer)(params, lr=args.learning_rate)
    
    # initialize best model info, and lr controller
    best_f1 = 0
    best_model = None 
    best_weighing_params = None
    lr_controller = LearningRateController()

    # load checkpoint, if exists
    args.start_epoch = 0 
    args.epoch_step = -1
    ckpt_path = os.path.join(
        args.model_path, args.model_name + '.ckpt'
    )
    if os.path.exists(ckpt_path):
        logger.info(f'Loading checkpoint from {ckpt_path}.')
        checkpoint = torch.load(ckpt_path)
        best_model = checkpoint['best_model']
        best_weighing_params = checkpoint['best_weighing_params']
        best_f1 = checkpoint['best_f1']
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_controller = checkpoint['lr_controller']
        args.start_epoch = checkpoint['epoch']
        args.epoch_step = checkpoint['step']

    assert best_model is not None
    model.load_state_dict(best_model)
    encoder.weighing_params.data = best_weighing_params.data
    model.eval()

    
    word_info = list()

    for idx, (sents, spans, labels) in enumerate(data_loader['test']):
        if idx == 7:
            break
        if torch.cuda.is_available():
            sents = sents.cuda()
            spans = spans.cuda()
            labels = labels.cuda()
        batch_ids = sents 
        input_mask = (batch_ids != encoder.tokenizer.pad_token_id).cuda().float()
        if 'spanbert' in encoder.model_name: 
            embeddings = encoder.model.embeddings(batch_ids, token_type_ids=torch.zeros_like(batch_ids)).requires_grad_()
            encoded_layers = encoder.model(batch_ids, attention_mask=input_mask)
            encoded_layers = [embedding_output] + encoded_layers
            last_layer_states = encoded_layers[-1]
        else:
            token_type_ids = torch.zeros_like(batch_ids)
            attention_mask = input_mask
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=next(encoder.model.parameters()).dtype) # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            head_mask = [None] * encoder.model.config.num_hidden_layers
            embeddings = encoder.model.embeddings(batch_ids, position_ids=None, token_type_ids=token_type_ids).requires_grad_()
            encoder_outputs = encoder.model.encoder(embeddings, extended_attention_mask, head_mask=head_mask)
            encoded_layers = encoder_outputs[1]
        wtd_encoded_repr = 0
        soft_weight = nn.functional.softmax(encoder.weighing_params, dim=0)
        for i in range(encoder.num_layers):
            wtd_encoded_repr += soft_weight[i] * encoded_layers[i]
        output = encoder.proj(wtd_encoded_repr) if encoder.proj else wtd_encoded_repr
        preds = model(output, spans[:, 0], spans[:, 1] - 1)
        loss = nn.BCELoss()(preds, labels.float())
        loss.backward()
        avg_inside_attn = total_instances = 0
        avg_spearman_r = 0
        attn_weights = embeddings.grad.abs().sum(-1) / embeddings.grad.abs().sum(-1).max(-1)[0].unsqueeze(-1)
        for i, sent in enumerate(sents):
            inside_attention = sum_attention = 0
            words = encoder.tokenizer.convert_ids_to_tokens(sent.tolist())
            for j in range(sent.shape[0]):
                if sent[j].item() == encoder.tokenizer.pad_token_id:
                    span_mask = [1 if spans[i][0].item() <= k < spans[i][1].item() else 0 for k in range(j)]
                    avg_spearman_r += spearmanr(attn_weights[i][:j].cpu().tolist(), span_mask)[0]
                    break
                else:
                    word_info.append(
                        {
                            'word': words[j],
                            'attention': float('{:.2f}'.format(1-attn_weights[i][j].item()))
                        }
                    )
                    sum_attention += attn_weights[i][j].item()
                    if spans[i][0].item() <= j < spans[i][1].item():
                        inside_attention += attn_weights[i][j].item()
            word_info.append(
                {
                    'word': '<br>',
                    'attention': 1.00
                }
            )
            for j in range(sent.shape[0]):
                if sent[j].item() == encoder.tokenizer.pad_token_id:
                    break
                else:
                    word_info.append(
                        {
                            'word': words[j],
                            'attention': 0.0 if spans[i][0].item() <= j < spans[i][1].item() else 1.00
                        }
                    )
            word_info.append({'word': '<br>','attention': 1.00})
            word_info.append({'word': '<br>','attention': 1.00})
            avg_inside_attn += inside_attention / sum_attention
            total_instances += 1
        del embeddings, attn_weights, loss, encoded_layers, sents, spans, labels
    
    # config viz path
    args.viz_file = os.path.join(args.viz_path, args.model_name + '.html')
    fout = open(args.viz_file, 'w')
    fout.write(HTML_TEMPLATE.format(json.dumps(word_info)).replace('LLLLB', '{').replace('RRRRB', '}') + '\n')
    fout.close()

    print('{:.2f}'.format(avg_inside_attn / total_instances * 100))
    print('{:.3f}'.format(avg_spearman_r / total_instances))
