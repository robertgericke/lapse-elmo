from argparse import ArgumentParser
from torch import cuda

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str, help='dataset directory')
    parser.add_argument('vocab', type=str, help='vocabulary file')
    # training options
    training = parser.add_argument_group('training')
    training.add_argument('-e', '--epochs', default=1, type=int, help='number of epochs to run')
    training.add_argument('-b', '--batch_size', default=128, type=int, help='number of sentences per batch')
    training.add_argument('-s', '--samples', default=8192, type=int, help='number of samples for softmax loss')
    training.add_argument('--no_cuda', default=False, dest='cuda', action='store_false', help='disable CUDA training')
    # elmo options
    elmo = parser.add_argument_group('elmo')
    elmo.add_argument('-ed', '--embedding_dim', default=512, type=int, help='dimension of the word embeddings')
    elmo.add_argument('-cs', '--cell_size', default=4096, type=int, help='dimension of the lstm memory cell')
    elmo.add_argument('-l', '--layers', default=2, type=int, help='number of lstm layers')
    elmo.add_argument('-rd', '--recurrent_dropout', default=0.1, type=float, help='recurrent dropout for each lstm')
    elmo.add_argument('-fd', '--dropout', default=0.1, type=float, help='final dropout for elmo representations')
    # distribution options
    distribution = parser.add_argument_group('distribution')
    distribution.add_argument('-ps', '--servers', default=1, type=int, help='number of servers')
    distribution.add_argument('-sd', '--sync_dense', default=1, type=int, help='synchronize dense model parameters every n steps')
    args = parser.parse_args()
    args.world_size = args.servers
    return args


if __name__ == "__main__":
    print(parse_arguments())