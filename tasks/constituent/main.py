import argparse
import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from encoders import Encoder
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


def validate(data_loader, model):
    encoder, model = model
    numerator = denom_p = denom_r = 0
    for sents, spans, labels in data_loader:
        if torch.cuda.is_available():
            sents = sents.cuda()
            spans = spans.cuda()
            labels = labels.cuda()
        # open range: [start_id, end_id)
        # using close range: [start_id, end_id - 1]
        output = encoder(sents)
        span_reprs = get_span_repr(
            output, spans[:, 0], spans[:, 1] - 1, method=args.encoding_method
        )
        preds = (model(span_reprs.detach()) > 0.5).long()
        num, dp, dr = instance_f1_info(labels, preds)
        numerator += num
        denom_p += dp
        denom_r += dr
    return f1_score(numerator, denom_p, denom_r)


def train(epoch, data_loader, model, actual_step, optimizer, 
        best_model_info, logger, lr_controller, args):
    terminate = False
    encoder, model = model
    model.train()
    best_f1, best_model = best_model_info
    cummulated_loss = cummulated_num = 0
    for step, (sents, spans, labels) in enumerate(data_loader['train']):
        if torch.cuda.is_available():
            sents = sents.cuda()
            spans = spans.cuda()
            labels = labels.cuda()
        with torch.no_grad():
            output = encoder(sents)
            span_reprs = get_span_repr(
                output, spans[:, 0], spans[:, 1] - 1, 
                method=args.encoding_method
            )
        preds = model(span_reprs.detach())
        # optimize model
        loss = nn.BCELoss()(preds, labels.float())
        loss.backward()
        optimizer.step()
        # update metadata
        cummulated_loss += loss.item() * sents.shape[0]
        cummulated_num += sents.shape[0]
        # log
        if (actual_step + step + 1) % args.log_step == 0:
            logger.info(
                f'Epoch #{epoch} | Step {actual_step + step + 1} | '
                f'loss {cummulated_loss / cummulated_num:8.4f}'
            )
        # validate
        if (actual_step + step + 1) % args.eval_step == 0:
            model.eval()
            logger.info('-' * 80)
            with torch.no_grad():
                curr_f1 = validate(data_loader['development'], (encoder, model))
            logger.info(
                f'Validation '
                f'F1 {curr_f1 * 100:6.2f}%'
            )
            if curr_f1 > best_f1:
                best_f1 = curr_f1
                best_model = model.state_dict()
                logger.info('New best model!')
            logger.info('-' * 80)
            model.train()
            # update validation result
            not_improved_epoch = lr_controller.add_value(curr_f1)
            if not_improved_epoch == 0:
                pass
            elif not_improved_epoch >= lr_controller.terminate_range:
                terminate = True
            elif not_improved_epoch % lr_controller.weight_decay_range == 0:
                logger.info(
                    f'Re-initialize learning rate to '
                    f'{optimizer.param_groups[0]["lr"] / 2.0:.8f}')
                optimizer = getattr(torch.optim, args.optimizer)(
                    model.parameters(), lr=optimizer.param_groups[0]['lr'] / 2.0
                )
    return best_f1, best_model, actual_step + step, optimizer, \
        lr_controller, terminate
        

if __name__ == '__main__':
    # arguments from snippets
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, 
        default='tasks/constituent/data/edges/ontonotes/const/nonterminal')
    parser.add_argument('--model-path', type=str, 
        default='tasks/constituent/checkpoints')
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--log-step', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=1000)
    # slurm supportive snippets 
    parser.add_argument('--epoch-run', type=int, default=5)
    # customized arguments
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256])
    parser.add_argument('--model', type=str, default='bert')
    parser.add_argument('--model-size', type=str, default='base')
    parser.add_argument('--uncased', action='store_false', dest='cased')
    parser.add_argument('--encoding-method', type=str, default='avg')
    args = parser.parse_args()

    # save arguments
    args_save_path = os.path.join(
        args.model_path, args.model_name + '.args.pt'
    )
    torch.save(args, args_save_path)

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
    logger.info('Creating datasets.')
    encoder = Encoder(args.model, args.model_size, args.cased)
    data_set = dict()
    data_loader = dict()
    for split in ['train', 'development', 'test']:
        data_set[split] = ConstituentDataset(
            os.path.join(args.data_path, f'{split}.json'),
            encoder=encoder
        )
        data_loader[split] = DataLoader(data_set[split], args.batch_size, 
            collate_fn=collate_fn, shuffle=(split=='train'))
    
    # initialize models: MLP
    logger.info('Initializing models.')
    if args.encoding_method in ['avg', 'max', 'diff']:
        input_dim = encoder.hidden_size
    elif args.encoding_method in ['diff_sum']:
        input_dim = encoder.hidden_size * 2
    elif args.encoding_method in ['coherent']:
        input_dim = encoder.hidden_size // 4 * 2 + 1
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
    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(), 
        lr=args.learning_rate
    )
    
    # initialize best model info, and lr controller
    actual_step = 0
    best_f1 = 0
    best_model = None
    lr_controller = LearningRateController()

    # load checkpoint, if exists
    args.start_epoch = 0
    ckpt_path = os.path.join(
        args.model_path, args.model_name + '.ckpt'
    )
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_model = checkpoint['best_model']
        best_f1 = checkpoint['best_f1']
        lr_controller = checkpoint['lr_controller']
        args.start_epoch = checkpoint['epoch']
        actual_step = checkpoint['step']

    # training
    terminate = False
    for epoch in range(args.start_epoch, 
            min(args.epochs, args.epoch_run + args.start_epoch)):
        best_f1, best_model, actual_step, optimizer, lr_controller, \
            terminate = train(
                epoch, data_loader, (encoder, model), actual_step, optimizer,
                (best_f1, best_model), logger, lr_controller, args
            )
        # save checkpoint 
        torch.save({
            'model': model.state_dict(),
            'best_model': best_model,
            'best_f1': best_f1,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'lr_controller': lr_controller,
            'step': actual_step
        }, ckpt_path)

    # finished training, testing
    if (args.start_epoch + args.epoch_run == args.epochs) or terminate:
        assert best_model is not None
        model.load_state_dict(best_model)
        model.eval()
        with torch.no_grad():
            test_f1 = validate(data_loader['test'], (encoder, model))
        logger.info(f'Test F1 {test_f1 * 100:6.2f}%')
