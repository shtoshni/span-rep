from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from dataset import load_data, ID_TO_LABEL

def main():
    data = load_data(Path('data'), mode='elmo', bio=True)
    _, labels, _ = data['dev']

    lengths = []
    for sent in labels:
        ent_len = None
        for tag in map(ID_TO_LABEL.get, sent):
            if tag == 'O' or tag[0] == 'B':
                if ent_len is not None:
                    lengths.append(ent_len)
                ent_len = None

            if tag[0] in ('I', 'B'):
                if ent_len is None:
                    ent_len = 0
                ent_len += 1

        if ent_len:
            lengths.append(ent_len)

    plt.figure(figsize=(11, 8))
    plt.title('NER entity length counts')
    plt.xlabel('Length')
    plt.ylabel('Occurrences')
    plt.yscale('log')

    plt.hist(lengths)
    plt.savefig('length_hist.png', dpi=300)

if __name__ == '__main__':
    main()
