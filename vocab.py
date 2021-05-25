from data import get_filenames

from argparse import ArgumentParser
from collections import Counter


def extract_vocab(dataset_path: str, vocab_path: str, threshold: int = 3):
    filenames = get_filenames(dataset_path)

    count = Counter()
    for filename in sorted(filenames):
        print("loading: " + filename)
        with open(filename, 'r') as file:
            for line in file:
                count.update([word for word in line.strip().split()])

    print("writing...")
    with open(vocab_path, "w") as file:
        for key, frequency in count.most_common():
            if frequency >= threshold:
                file.write(key + '\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str, help='dataset directory')
    parser.add_argument('-v', '--vocab', type=str, default="vocab.txt", help='vocabulary filename')
    parser.add_argument('-t', '--threshold', default=3, type=int, help='min threshold for occurrence')
    args = parser.parse_args()
    extract_vocab(args.dataset, args.vocab, args.threshold)
