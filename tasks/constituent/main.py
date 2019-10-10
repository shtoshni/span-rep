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


def validate(data_loader, model):
    encoder, model = model
    total_correct_num = total_num = 0
    for sents, spans, labels in data_loader:
        # open range: [start_id, end_id); close range: [start_id, end_id - 1]
        output = encoder(sents)
        span_reprs = get_span_repr(
            output, spans[:, 0], spans[:, 1] - 1, method=args.encoding_method
        )
        preds = model(span_reprs.detach())
        labels = torch.tensor(labels)
        if torch.cuda.is_available():
            labels = labels.cuda()
        correct_num = (preds.max(1)[1] == labels).long().sum()
        total_correct_num += correct_num.item()
        total_num += sents.shape[0]
    return float(total_correct_num) / total_num


def train(epoch, data_loader, model, optimizer, best_model_info, logger, args):
    encoder, model = model
    best_acc, best_model = best_model_info
    cummulated_loss = cummulated_correct = cummulated_num = 0
    eval_step = len(data_loader['train']) // 5
    for step, (sents, spans, labels) in enumerate(data_loader['train']):
        with torch.no_grad():
            output = encoder(sents)
            span_reprs = get_span_repr(
                output, spans[:, 0], spans[:, 1] - 1, 
                method=args.encoding_method
            )
        preds = model(span_reprs.detach())
        labels = torch.tensor(labels)
        if torch.cuda.is_available():
            labels = labels.cuda()
        # optimize model
        loss = nn.CrossEntropyLoss()(preds, labels)
        loss.backward()
        optimizer.step()
        # update metadata
        correct_num = (preds.max(1)[1] == labels).long().sum()
        cummulated_loss += loss.item() * sents.shape[0]
        cummulated_correct += correct_num.item()
        cummulated_num += sents.shape[0]
        # log
        if (step + 1) % args.log_step == 0:
            logger.info(
                f'Epoch #{epoch} | Step {step+1} | '
                f'loss {cummulated_loss / cummulated_num:8.4f} | '
                f'acc. {cummulated_correct / cummulated_num * 100:6.2f}%'
            )
        # validate
        if (step + 1) % eval_step == 0:
            logger.info('-' * 80)
            with torch.no_grad():
                curr_acc = validate(data_loader['development'], (encoder, model))
            logger.info(
                f'Validation '
                f'acc. {curr_acc * 100:6.2f}%'
            )
            if curr_acc > best_acc:
                best_acc = curr_acc
                best_model = model.state_dict()
                logger.info('New best model!')
            logger.info('-' * 80)
    return best_acc, best_model
        

if __name__ == '__main__':
    # arguments from snippets
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, 
        default='tasks/constituent/data/edges/ontonotes/const/nonterminal')
    parser.add_argument('--model-path', type=str, 
        default='tasks/constituent/checkpoints')
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--log-step', type=int, default=100)
    # slurm supportive snippets 
    parser.add_argument('--epoch-run', type=int, default=2)
    # customized arguments
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[1024])
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
            collate_fn=collate_fn)
        # shuffle=(split=='train')) # TODO(freda) shuffle for training set
    
    # initialize models: MLP
    logger.info('Initializing models.')
    if args.encoding_method in ['avg', 'max', 'diff']:
        input_dim = encoder.hidden_size
    elif args.encoding_method in ['diff_sum']:
        input_dim = encoder.hidden_size * 2
    elif args.encoding_method in ['coherent']:
        input_dim = encoder.hidden_size // 4 * 2 + 1
    else:
        raise Exception(f'Encoding method {args.encoding_method} not supported.')
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
    
    # load checkpoint, if exists
    args.start_epoch = 0
    ckpt_path = os.path.join(
        args.model_path, args.model_name + '.ckpt'
    )
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch']

    # training
    best_acc = 0
    best_model = None
    for epoch in range(args.start_epoch, 
            min(args.epochs, args.epoch_run + args.start_epoch)):
        best_acc, best_model = train(epoch, data_loader, (encoder, model), 
            optimizer, (best_acc, best_model), logger, args)
        # save checkpoint 
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1
        }, ckpt_path)

    # finished training, testing
    if args.start_epoch + args.epoch_run == args.epochs:
        assert best_model is not None
        model.load_state_dict(best_model)
        with torch.no_grad():
            test_acc = validate(data_loader['test'], (encoder, model))
        logger.info(f'Test acc. {curr_acc * 100:6.2f}%')
