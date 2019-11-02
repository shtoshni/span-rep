import argparse
import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from encoders.pretrained_transformers import Encoder
from encoders.pretrained_transformers.batched_span_reprs import get_span_repr
from tasks.constituent.data import ConstituentDataset, collate_fn
from tasks.constituent.models import MultiLayerPerceptron
from tasks.constituent.utils import instance_f1_info, f1_score


class LearningRateController(object):
    def __init__(self, weight_decay_range=5, terminate_range=20):
        self.data = list()
        self.not_improved = 0
        self.weight_decay_range = weight_decay_range
        self.terminate_range = terminate_range
        self.best_performance = -1e10
    
    def add_value(self, val):
        # add value 
        if len(self.data) == 0 or val > self.best_performance:
            self.not_improved = 0
            self.best_performance = val
        else:
            self.not_improved += 1
        self.data.append(val)
        return self.not_improved


def forward_batch(model, data, valid=False):
    sents, spans, labels = data
    with torch.no_grad():
        output = encoder(sents)
        span_reprs = get_span_repr(
            output, spans[:, 0], spans[:, 1] - 1, 
            method=args.encoding_method
        )
    preds = model(span_reprs.detach())
    if valid:
        return preds
    else:
        loss = nn.BCELoss()(preds, labels.float())
        return preds, loss


def validate(loader, model):
    # save the random state for recovery
    rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.random.get_rng_state()
    numerator = denom_p = denom_r = 0
    for sents, spans, labels in loader:
        if torch.cuda.is_available():
            sents = sents.cuda()
            spans = spans.cuda()
            labels = labels.cuda()
        preds = forward_batch(model, (sents, spans, labels), True)
        pred_labels = (preds > 0.5).long()
        num, dp, dr = instance_f1_info(labels, pred_labels)
        numerator += num
        denom_p += dp
        denom_r += dr
    # recover the random state
    torch.random.set_rng_state(rng_state)
    torch.cuda.random.set_rng_state(cuda_rng_state)
    return f1_score(numerator, denom_p, denom_r)


def analysis(loader, model): 
    import numpy as np  
    numerator = np.zeros(1000) 
    denom_p = np.zeros(1000) 
    denom_r = np.zeros(1000) 
    for sents, spans, labels in loader: 
        if torch.cuda.is_available(): 
            sents = sents.cuda() 
            spans = spans.cuda() 
            labels = labels.cuda() 
        preds = forward_batch(model, (sents, spans, labels), True) 
        pred_labels = (preds > 0.5).long() 
        eqs = pred_labels * labels 
        for i in range(spans.shape[0]): 
            length = (spans[i][1] - spans[i][0] + 1).item() 
            num = eqs[i].sum() 
            dp = pred_labels[i].sum() 
            dr = labels[i].sum() 
            numerator[length] += num 
            denom_p[length] += dp 
            denom_r[length] += dr 
    return numerator, denom_p, denom_r


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
    if args.encoding_method in ['avg', 'max', 'diff', 'attn']:
        input_dim = args.proj_dim if args.use_proj else encoder.hidden_size
    elif args.encoding_method in ['diff_sum']:
        input_dim = (
            args.proj_dim if args.use_proj else encoder.hidden_size) * 2
    elif args.encoding_method in ['coherent']:
        input_dim = (args.proj_dim if args.use_proj else encoder.hidden_size)\
            // 4 * 2 + 1
    else:
        raise Exception(
            f'Encoding method {args.encoding_method} not supported.')
    model = MultiLayerPerceptron(
        input_dim, args.hidden_dims, len(ConstituentDataset.label_dict)
    )
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        model = model.cuda()
    
    # initialize optimizer
    logger.info('Initializing optimizer.')
    params = list(model.parameters())
    if args.use_proj:
        params += list(encoder.proj.parameters())
    optimizer = getattr(torch.optim, args.optimizer)(
        params, lr=args.learning_rate
    )
    
    # initialize best model info, and lr controller
    best_f1 = 0
    best_model = None
    best_proj = None
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
        model.load_state_dict(checkpoint['model'])
        best_model = checkpoint['best_model']
        best_proj = checkpoint['best_proj']
        best_f1 = checkpoint['best_f1']
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_controller = checkpoint['lr_controller']
        torch.cuda.random.set_rng_state(checkpoint['cuda_rng_state'])
        args.start_epoch = checkpoint['epoch']
        args.epoch_step = checkpoint['step']
        if args.use_proj:
            encoder.proj.load_state_dict(checkpoint['enc_proj'])
        from IPython import embed; embed(using=False)
        if lr_controller.not_improved >= lr_controller.terminate_range:
            logger.info('No more optimization, exiting.')
            exit(0)

    # training
    terminate = False
    for epoch in range(args.epochs):
        if terminate:
            break
        model.train()
        cummulated_loss = cummulated_num = 0
        for step, (sents, spans, labels) in enumerate(data_loader['train']):
            if terminate:
                break
            # ignore batches to recover the same data loader state of checkpoint
            if (epoch < args.start_epoch) or (epoch == args.start_epoch and \
                    step <= args.epoch_step):
                continue
            if torch.cuda.is_available():
                sents = sents.cuda()
                spans = spans.cuda()
                labels = labels.cuda()
            preds, loss = forward_batch(model, (sents, spans, labels))
            # optimize model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update metadata
            cummulated_loss += loss.item() * sents.shape[0]
            cummulated_num += sents.shape[0]
            # log
            actual_step = len(data_loader['train']) * epoch + step + 1
            if actual_step % args.log_step == 0:
                logger.info(
                    f'Train '
                    f'Epoch #{epoch} | Step {actual_step} | '
                    f'loss {cummulated_loss / cummulated_num:8.4f}'
                )
            # validate
            if actual_step % args.eval_step == 0:
                model.eval()
                logger.info('-' * 80)
                with torch.no_grad():
                    curr_f1 = validate(data_loader['development'], model)
                logger.info(
                    f'Validation '
                    f'F1 {curr_f1 * 100:6.2f}%'
                )
                # update when there is a new best model
                if curr_f1 > best_f1:
                    best_model_path = os.path.join(
                        args.model_path, args.model_name + '.best.pt'
                    )
                    best_f1 = curr_f1
                    best_model = model.state_dict()
                    if args.use_proj:
                        best_proj = encoder.proj.state_dict()
                        torch.save((best_model, best_proj), best_model_path)
                    else:
                        torch.save(best_model, best_model_path)
                    logger.info('New best model!')
                logger.info('-' * 80)
                model.train()
                # save checkpoint
                torch.save({
                    'model': model.state_dict(),
                    'best_proj': best_proj,
                    'best_model': best_model,
                    'best_f1': best_f1,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'lr_controller': lr_controller,
                    'cuda_rng_state': torch.cuda.random.get_rng_state(),
                    'enc_proj': None if not args.use_proj else \
                        encoder.proj.state_dict()
                }, ckpt_path)
                # update validation result
                not_improved_epoch = lr_controller.add_value(curr_f1)
                if not_improved_epoch == 0:
                    pass
                elif not_improved_epoch >= lr_controller.terminate_range:
                    logger.info('Terminating due to lack of validation improvement.')
                    terminate = True
                elif not_improved_epoch % lr_controller.weight_decay_range == 0:
                    logger.info(
                        f'Re-initialize learning rate to '
                        f'{optimizer.param_groups[0]["lr"] / 2.0:.8f}'
                    )
                    optimizer = getattr(torch.optim, args.optimizer)(
                        params, lr=optimizer.param_groups[0]['lr'] / 2.0
                    )
                if (time.time() - args.start_time) >= args.time_limit:
                    logger.info('Training time is almost up -- terminating.')
                    exit(0)

    # finished training, testing
    assert best_model is not None
    assert (not args.use_proj) or (best_proj is not None)
    model.load_state_dict(best_model)
    if args.use_proj:
        encoder.proj.load_state_dict(best_proj)
    model.eval()
    with torch.no_grad():
        test_f1 = validate(data_loader['test'], model)
    logger.info(f'Test F1 {test_f1 * 100:6.2f}%')
