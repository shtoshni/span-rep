import torch
import argparse
from model import TaskModel
from data import TaskDataset
import logging
from collections import OrderedDict
import hashlib
from os import path
import os
import sys

from encoders.pretrained_transformers import Encoder


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

MAX_STUCK_EVALS = 20


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-data_dir", type=str,
        default="/home/shtoshni/Research/hackathon_2019/tasks/mention_detection/data/mention")
    parser.add_argument(
        "-model_dir", type=str,
        default="/home/shtoshni/Research/hackathon_2019/tasks/mention_detection/checkpoints")
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-eval_batch_size", type=int, default=64)
    parser.add_argument("-eval_steps", type=int, default=1000)
    parser.add_argument("-n_epochs", type=int, default=20)
    parser.add_argument("-lr", type=float, default=5e-4)
    parser.add_argument("-span_dim", type=int, default=256)
    parser.add_argument("-model", type=str, default="bert")
    parser.add_argument("-model_size", type=str, default="base")
    parser.add_argument("-pool_method", default="avg", type=str)
    parser.add_argument("-train_frac", default=1.0, type=float,
                        help="Can reduce this for quick testing.")
    parser.add_argument("-seed", type=int, default=0, help="Random seed")
    parser.add_argument("-eval", default=False, action="store_true")
    parser.add_argument('-slurm_id', help="Slurm ID",
                        default=None, type=str)

    hp = parser.parse_args()
    return hp


def save_model(model, optimizer, scheduler, steps_done, max_f1, num_stuck_evals, location):
    """Save model."""
    save_dict = {}
    save_dict['weighing_params'] = model.encoder.weighing_params
    save_dict['span_net'] = model.span_net.state_dict()
    save_dict['label_net'] = model.label_net.state_dict()
    save_dict.update({
        'steps_done': steps_done,
        'max_f1': max_f1,
        'num_stuck_evals': num_stuck_evals,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'rng_state': torch.get_rng_state(),
    })
    torch.save(save_dict, location)
    logging.info("Model saved at: %s" % (location))


def train(model, train_iter, val_iter, optimizer, scheduler,
          model_dir, best_model_dir, init_steps=0, eval_steps=1000, num_steps=120000, max_f1=0,
          init_num_stuck_evals=0):
    model.train()

    steps_done = init_steps
    num_stuck_evals = init_num_stuck_evals
    while (steps_done < num_steps) and (num_stuck_evals < 20):
        logging.info("Epoch started")
        for idx, batch_data in enumerate(train_iter):
            optimizer.zero_grad()
            loss = model(batch_data)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0)

            optimizer.step()
            steps_done += 1

            if (steps_done % eval_steps) == 0:
                logging.info("Evaluating at %d" % steps_done)
                f1, _ = eval(model, val_iter)
                # Scheduler step
                scheduler.step(f1)

                if f1 > max_f1:
                    num_stuck_evals = 0
                    max_f1 = f1
                    logging.info("Max F1: %.3f" % max_f1)
                    location = path.join(best_model_dir, "model.pt")
                    save_model(model, optimizer, scheduler, steps_done, max_f1, num_stuck_evals,
                               location)
                else:
                    num_stuck_evals += 1

                location = path.join(model_dir, "model.pt")
                save_model(model, optimizer, scheduler, steps_done, max_f1, num_stuck_evals, location)

                logging.info("Val F1: %.3f Steps: %d (Max F1: %.3f)" % (f1, steps_done, max_f1))

                if num_stuck_evals >= MAX_STUCK_EVALS:
                    logging.info(f'No improvement for {MAX_STUCK_EVALS} evaluations')
                    break

                sys.stdout.flush()

        logging.info("Epoch done!\n")

    logging.info("Training done!\n")


def eval(model, val_iter, final_eval=False):
    model.eval()
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    all_res = []
    with torch.no_grad():
        for batch_data in val_iter:
            _, pred, label = model(batch_data)
            pred = (pred > 0.5).float()

            tp += torch.sum(label * pred)
            tn += torch.sum((1 - label) * (1 - pred))
            fp += torch.sum((1 - label) * pred)
            fn += torch.sum(label * (1 - pred))

            batch_size = label.shape[0]
            span = batch_data.span
            for idx in range(batch_size):
                if final_eval:
                    all_res.append({'span': span[idx, :].tolist(),
                                    'pred': pred[idx],
                                    'label': label[idx],
                                    'corr': pred[idx] == label[idx]})

    if tp > 0:
        recall = tp/(tp + fn)
        precision = tp/(tp + fp)

        f_score = (2 * recall * precision) / (recall + precision)
    else:
        f_score = 0.0

    model.train()
    return f_score, all_res


def get_model_name(hp):
    opt_dict = OrderedDict()
    # Only include important options in hash computation
    imp_opts = ['model', 'model_size', 'batch_size', 'eval_steps',
                'n_epochs',  'span_dim', 'pool_method', 'train_frac',
                'seed', 'lr']
    hp_dict = vars(hp)
    for key in imp_opts:
        val = hp_dict[key]
        opt_dict[key] = val
        logging.info("%s\t%s" % (key, val))

    str_repr = str(opt_dict.items())
    hash_idx = hashlib.md5(str_repr.encode("utf-8")).hexdigest()
    model_name = "mention_detection_" + str(hash_idx)
    return model_name


def write_res(all_res, output_file):
    with open(output_file, 'w') as f:
        f.write('span_width\tlabel\tcorr\n')
        for res in all_res:
            span, label, corr = (res['span'], res['label'], res['corr'])
            # End points of the spans are included, hence the +1 in width calc
            span_width = span[1] - span[0] + 1
            f.write('%d\t%d\t%d\n' % (span_width, label, corr))


def final_eval(hp, model, best_model_dir, val_iter, test_iter):
    location = path.join(best_model_dir, "model.pt")
    model_dir = path.dirname(best_model_dir)
    val_f1, test_f1 = 0, 0
    if path.exists(location):
        checkpoint = torch.load(location)
        model.span_net.load_state_dict(checkpoint['span_net'])
        model.label_net.load_state_dict(checkpoint['label_net'])
        model.encoder.weighing_params = checkpoint['weighing_params']
        val_f1, val_res = eval(model, val_iter, final_eval=True)
        val_file = path.join(model_dir, "val_log.tsv")
        write_res(val_res, val_file)

        test_f1, test_res = eval(model, test_iter, final_eval=True)
        test_file = path.join(model_dir, "test_log.tsv")
        write_res(test_res, test_file)

        logging.info("Val F1: %.3f" % val_f1)
        logging.info("Test F1: %.3f" % test_f1)
    return (val_f1, test_f1)


def main():
    hp = parse_args()

    # Setup model directories
    model_name = get_model_name(hp)
    model_path = path.join(hp.model_dir, model_name)
    best_model_path = path.join(model_path, 'best_models')
    if not path.exists(model_path):
        os.makedirs(model_path)
    if not path.exists(best_model_path):
        os.makedirs(best_model_path)

    # Set random seed
    torch.manual_seed(hp.seed)

    # Hacky way of assigning the number of labels.
    encoder = Encoder(model=hp.model, model_size=hp.model_size, fine_tune=False,
                      cased=False)
    # Load data
    logging.info("Loading data")
    train_iter, val_iter, test_iter = TaskDataset.iters(
        hp.data_dir, encoder, batch_size=hp.batch_size,
        eval_batch_size=hp.eval_batch_size, train_frac=hp.train_frac)
    logging.info("Data loaded")

    # Initialize the model
    model = TaskModel(encoder, **vars(hp)).cuda()
    sys.stdout.flush()

    optimizer = torch.optim.Adam(model.get_other_params(), lr=hp.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, verbose=True)
    steps_done = 0
    max_f1 = 0
    init_num_stuck_evals = 0
    num_steps = (hp.n_epochs * len(train_iter.data())) // hp.batch_size
    # Quantize the number of training steps to eval steps
    num_steps = (num_steps // hp.eval_steps) * hp.eval_steps
    logging.info("Total training steps: %d" % num_steps)

    location = path.join(model_path, "model.pt")
    if path.exists(location):
        logging.info("Loading previous checkpoint")
        checkpoint = torch.load(location)
        model.encoder.weighing_params = checkpoint['weighing_params']
        model.span_net.load_state_dict(checkpoint['span_net'])
        model.label_net.load_state_dict(checkpoint['label_net'])
        optimizer.load_state_dict(
            checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(
            checkpoint['scheduler_state_dict'])
        steps_done = checkpoint['steps_done']
        init_num_stuck_evals = checkpoint['num_stuck_evals']
        max_f1 = checkpoint['max_f1']
        torch.set_rng_state(checkpoint['rng_state'])
        logging.info("Steps done: %d, Max F1: %.3f" % (steps_done, max_f1))

    if not hp.eval:
        train(model, train_iter, val_iter, optimizer, scheduler,
              model_path, best_model_path, init_steps=steps_done, max_f1=max_f1,
              eval_steps=hp.eval_steps, num_steps=num_steps,
              init_num_stuck_evals=init_num_stuck_evals)

    val_f1, test_f1 = final_eval(hp, model, best_model_path, val_iter, test_iter)
    perf_dir = path.join(hp.model_dir, "perf")
    if not path.exists(perf_dir):
        os.makedirs(perf_dir)
    if hp.slurm_id:
        perf_file = path.join(perf_dir, hp.slurm_id + ".txt")
    else:
        perf_file = path.join(model_path, "perf.txt")
    with open(perf_file, "w") as f:
        f.write("%s\n" % (model_path))
        f.write("%s\t%.4f\n" % ("Valid", val_f1))
        f.write("%s\t%.4f\n" % ("Test", test_f1))


if __name__ == '__main__':
    main()
