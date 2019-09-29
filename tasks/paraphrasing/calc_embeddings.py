import os
from os import path
from encoder import Encoder
import torch
from span_reprs import get_avg_repr, get_diff_repr, \
    get_max_pooling_repr, get_alternate_repr

root_dir = "/home/shtoshni/Downloads/ppdb"
ppdb_file = path.join(root_dir, "ppdb_all.txt")

output_dir = path.join(root_dir, "outputs")
# Create a folder for all outputs
output_dir = path.join(root_dir, "outputs")
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
    for model_type in ['base', 'large']:
        # Initialize the encoder
        encoder = Encoder(model=model, model_type=model_type).cuda()
        span_repr_dict = {"avg": [], "diff": [], "max": [], "alternate": []}

        for sent in sent_list:
            token_ids = encoder.tokenize_sentence(sent)
            token_ids = token_ids.cuda()  # 1 x L
            hidden_states = encoder(token_ids, just_last_layer=True)
            # Ignore [CLS] and [SEP]
            start_idx = 1
            end_idx = token_ids.shape[1] - 2

            span_repr_dict["avg"].append(
                torch.squeeze(get_avg_repr(hidden_states, start_idx, end_idx),
                              dim=0).cpu().detach().numpy())

            span_repr_dict["diff"].append(
                torch.squeeze(get_diff_repr(hidden_states, start_idx, end_idx),
                              dim=0).cpu().detach().numpy())

            span_repr_dict["max"].append(
                torch.squeeze(get_max_pooling_repr(hidden_states, start_idx, end_idx), dim=0)
                .cpu().detach().numpy())

            span_repr_dict["alternate"].append(
                torch.squeeze(get_alternate_repr(hidden_states, start_idx, end_idx), dim=0)
                .cpu().detach().numpy())

        # Write to file
        for method, span_repr_list in span_repr_dict.items():
            file_prefix = (model + '-' + model_type + '-' + method)
            output_file = path.join(output_dir, file_prefix + ".tsv")

            with open(output_file, 'w') as f:
                for span_repr in span_repr_list:
                    f.write("\t".join(str(elem) for elem in span_repr))
                    f.write("\n")
