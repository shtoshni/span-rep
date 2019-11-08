import argparse 
import json
import numpy as np
from tqdm import tqdm 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--seed', type=int, default=1111)
    args = parser.parse_args()

    np.random.seed(args.seed)

    lines = open(args.input).readlines()
    fout = open(args.output, 'w')

    for line in tqdm(lines):
        info = json.loads(line.strip())
        length = len(info['text'].split())
        true_spans = set()
        for item in info['targets']:
            if item['span1'][0] != 0 or item['span1'][1] != length:  # discard the whole-sentence spans
                true_spans.add(tuple(item['span1']))
        sample_spans = dict()
        for i in range(length+1):
            sample_spans[i] = list()
        for l in range(length):
            for r in range(l+1, length+1):
                if (l, r) not in true_spans:
                    sample_spans[r-l].append((l,r))
        false_spans = list()
        for span in true_spans:
            span_length = span[1] - span[0]
            try:
                false_spans.append(sample_spans[span_length][np.random.randint(len(sample_spans[span_length]))])
            except ValueError:
                assert span_length == 1
        false_spans = [{'span1': item, 'label': 0} for item in false_spans]
        true_spans = [{'span1': item, 'label': 1} for item in true_spans] 
        info['targets'] = true_spans + false_spans
        fout.write(json.dumps(info) + '\n')

    fout.close()