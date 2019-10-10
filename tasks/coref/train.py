import torch
import argparse
from model import CorefModel
from data import CorefDataset
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-data_dir", type=str,
        default="/share/data/lang/users/freda/codebase/hackathon_2019/tasks/constituent/"
        "data/edges/ontonotes/coref")
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-n_epochs", type=int, default=10)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-lr_tune", type=float, default=1e-5)
    parser.add_argument("-span_dim", type=int, default=256)
    parser.add_argument("-model", type=str, default="bert")
    parser.add_argument("-model_size", type=str, default="base")
    parser.add_argument("-fine_tune", default=False, action="store_true")
    parser.add_argument("-pool_method", default="avg", type=str)
    parser.add_argument("-feedback", default=False, action="store_true",
                        help="Small setup to get quick feedback.")
    parser.add_argument("-seed", type=int, default=0, help="Random seed")

    hp = parser.parse_args()

    return hp


def train(model, train_iter, val_iter, optimizer, optimizer_tune, n_epochs):
    model.train()
    for epoch_idx in range(n_epochs):
        print("Start of epoch: %d" % (epoch_idx + 1))
        for idx, batch_data in enumerate(train_iter):
            optimizer.zero_grad()
            loss = model(batch_data)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0)

            optimizer.step()

            if optimizer_tune:
                optimizer_tune.step()

            if ((idx + 1) % 1000) == 0:
                f1 = eval(model, val_iter)
                logging.info(
                    "Val F1: %.3f Epoch: %d Steps: %d Loss: %.3f" %
                    (f1, epoch_idx + 1, idx + 1, loss.item()))


def eval(model, val_iter):
    model.eval()

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    eps = 1e-8
    with torch.no_grad():
        for batch_data in val_iter:
            label = batch_data.label.cuda().float()
            _, pred = model(batch_data)
            pred = (pred >= 0.5).float()

            tp += torch.sum(label * pred)
            tn += torch.sum((1 - label) * (1 - pred))
            fp += torch.sum((1 - label) * pred)
            fn += torch.sum(label * (1 - pred))

    recall = tp/(tp + fn + eps)
    precision = tp/(tp + fp + eps)

    f_score = (2 * recall * precision) / (recall + precision)
    model.train()
    return f_score


def main():
    hp = parse_args()

    # Set random seed
    torch.manual_seed(hp.seed)

    # Initialize the model
    model = CorefModel(**vars(hp)).cuda()

    # Load data
    logging.info("Loading data")
    train_iter, val_iter, test_iter = CorefDataset.iters(
        hp.data_dir, model.encoder, batch_size=hp.batch_size, feedback=hp.feedback)
    logging.info("Data loaded")

    optimizer_tune = None
    if hp.fine_tune:
        optimizer_tune = torch.optim.Adam(model.encoder.parameters(), lr=hp.lr_tune)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)

    train(model, train_iter, val_iter, optimizer, optimizer_tune, hp.n_epochs)
    print("Test F1: %.3f" % eval(model, test_iter))


if __name__ == '__main__':
    main()
