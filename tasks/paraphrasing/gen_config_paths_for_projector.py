from os import path
import json
import argparse

from calc_correlation_with_human_annotations import load_embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-root_dir", default="/home/shtoshni/Downloads/ppdb", type=str)
    parser.add_argument("-split", default="all", type=str)
    args = parser.parse_args()

    BASE_URL = "https://raw.githubusercontent.com/shtoshni92/projector-tensorflow/master/ppdb_all/"
    META_URL = BASE_URL + "metadata_file.txt"
    # BASE_URL = "https://ttic.uchicago.edu/~shtoshni/phrase_reps/"

    root_dir = args.root_dir

    info_dict = {}
    method_list = ["avg", "diff", "max", "diff_sum", "coherent"]
    emb_dir = path.join(root_dir, "outputs_" + args.split)

    for model in ['bert', 'spanbert', 'roberta', 'xlnet']:
        for model_size in ['base', 'large']:
            for method in method_list:
                full_emb_name = model + "-" + model_size + "-" + method
                info_dict[full_emb_name] = {}
                info_dict[full_emb_name]["tensorName"] = (
                    f'Model:{model} Size:{model_size} Span Method:{method}')

                emb_file = path.join(emb_dir, full_emb_name + ".tsv")
                emb_list = load_embeddings(emb_file)
                info_dict[full_emb_name]["tensorShape"] = [len(emb_list), len(emb_list[0])]
                info_dict[full_emb_name]["tensorPath"] = (BASE_URL + full_emb_name + ".tsv")
                info_dict[full_emb_name]["metadataPath"] = (META_URL)

    for model in ['gpt2']:
        for model_size in ['small', 'medium', 'large']:
            for method in method_list:
                full_emb_name = model + "-" + model_size + "-" + method
                info_dict[full_emb_name] = {}
                info_dict[full_emb_name]["tensorName"] = (
                    f'Model:{model} Size:{model_size} Span Method:{method}')

                emb_file = path.join(emb_dir, full_emb_name + ".tsv")
                emb_list = load_embeddings(emb_file)
                info_dict[full_emb_name]["tensorShape"] = [len(emb_list), len(emb_list[0])]
                info_dict[full_emb_name]["tensorPath"] = (BASE_URL + full_emb_name + ".tsv")
                info_dict[full_emb_name]["metadataPath"] = (META_URL)

    for model_name in info_dict:
        json_dump_file = path.join(emb_dir, "projector-" + model_name + ".json")
        with open(json_dump_file, "w") as dump_f:
            emb_dict = {"embeddings": [info_dict[model_name]]}
            dump_f.write(json.dumps(emb_dict, indent=2))
