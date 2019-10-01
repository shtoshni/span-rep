import os
import math
from os import path
from encoder import Encoder
import argparse


def get_sentence_reprs(sent_list, model, model_size):
    encoder = Encoder(model=model, model_size=model_size).cuda()
    span_repr_dict = {"avg": [], "diff": [], "max": [], "diff_sum": [], "coherent": []}

    batch_size = 32
    num_batches = math.ceil(len(sent_list)/batch_size)
    for chunk_idx in range(num_batches):
        batch_sents = sent_list[chunk_idx * batch_size: (chunk_idx + 1) * batch_size]
        token_ids, sent_len = encoder.tokenize_batch(batch_sents)  # B x L
        hidden_states = encoder(token_ids, just_last_layer=True)

        for method in span_repr_dict:
            batch_span_repr = encoder.get_sentence_repr(
                hidden_states, sent_len, method=method).tolist()
            span_repr_dict[method] += batch_span_repr

    # Write to file
    for method, span_repr_list in span_repr_dict.items():
        file_prefix = (model + '-' + model_size + '-' + method)
        output_file = path.join(output_dir, file_prefix + ".tsv")

        with open(output_file, 'w') as f:
            for span_repr in span_repr_list:
                f.write("\t".join(str(elem) for elem in span_repr))
                f.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-root_dir", default="/home/shtoshni/Downloads/ppdb", type=str)
    parser.add_argument("-split", default="all", type=str)
    args = parser.parse_args()

    root_dir = args.root_dir
    ppdb_file = path.join(root_dir, "ppdb_" + args.split + ".txt")
    # Create a folder for all outputs
    output_dir = path.join(root_dir, "outputs_" + args.split)
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    # First read all the sentences
    metadata_file = path.join(output_dir, "metadata_file.txt")
    sent_list = []
    with open(ppdb_file) as reader, open(metadata_file, 'w') as writer:
        for line in reader:
            sent1, sent2, _ = line.strip().split("|||")
            sent_list += [sent1, sent2]

            writer.write(sent1 + "\n")
            writer.write(sent2 + "\n")

    for model in ['bert', 'spanbert', 'roberta']:
        for model_size in ['base', 'large']:
            get_sentence_reprs(sent_list, model, model_size)

    for model in ['gpt2']:
        for model_size in ['small', 'medium', 'large']:
            get_sentence_reprs(sent_list, model, model_size)
