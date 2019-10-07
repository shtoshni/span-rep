import numpy as np
from os import path
import scipy.spatial as spatial
import scipy.stats as stats
import argparse


def load_embeddings(tsv_file):
    embedding_list = []
    with open(tsv_file) as f:
        for line in f:
            embedding_list.append(np.fromstring(line.strip(), sep='\t'))
    return embedding_list


def load_sentence_pairs(ppdb_file):
    pair_score_list = []
    with open(ppdb_file) as f:
        for line in f:
            sent1, sent2, score = line.strip().split('|||')
            pair_score_list.append((sent1, sent2, float(score)))

    return pair_score_list


def calculate_corr(pair_score_list, embedding_list):
    score_list = []
    sim_list = []
    for idx, (_, _, score) in enumerate(pair_score_list):
        emb1 = embedding_list[2 * idx]
        emb2 = embedding_list[2 * idx + 1]
        cos_sim = 1 - spatial.distance.cosine(emb1, emb2)

        sim_list.append(cos_sim)
        score_list.append(score)

    corr = stats.pearsonr(np.asarray(score_list), np.asarray(sim_list)/5.0)
    return corr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-root_dir", default="/home/shtoshni/Downloads/ppdb", type=str)
    parser.add_argument("-split", default="all", type=str)
    args = parser.parse_args()

    root_dir = args.root_dir
    ppdb_file = path.join(root_dir, "ppdb_" + args.split + ".txt")
    pair_score_list = load_sentence_pairs(ppdb_file)

    method_list = ["avg", "diff", "max", "diff_sum", "coherent"]
    emb_dir = path.join(root_dir, "outputs_" + args.split)
    for model in ['bert', 'spanbert', 'roberta', 'xlnet']:
        for model_size in ['base', 'large']:
            for method in method_list:
                file_prefix = (model + '-' + model_size + '-' + method)
                emb_file = path.join(emb_dir, file_prefix + ".tsv")
                emb_list = load_embeddings(emb_file)
                corr = calculate_corr(pair_score_list, emb_list)
                print("%s\t%s\t%s\t%.3f" % (model, model_size, method, corr[0]))

    for model in ['gpt2']:
        for model_size in ['small', 'medium', 'large']:
            for method in method_list:
                file_prefix = (model + '-' + model_size + '-' + method)
                emb_file = path.join(emb_dir, file_prefix + ".tsv")
                emb_list = load_embeddings(emb_file)
                corr = calculate_corr(pair_score_list, emb_list)
                print("%s\t%s\t%s\t%.3f" % (model, model_size, method, corr[0]))
