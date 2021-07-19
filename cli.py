from argparse import ArgumentParser
from torch import cuda

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str, help='dataset directory')
    parser.add_argument('vocab', type=str, help='vocabulary file')
    parser.add_argument('-ts', '--testset', type=str, help='directory of testdata')
    # training options
    training = parser.add_argument_group('training')
    training.add_argument('-e', '--epochs', default=2, type=int, help='number of epochs to run')
    training.add_argument('-b', '--batch_size', default=128, type=int, help='number of sentences per batch')
    training.add_argument('-s', '--samples', default=8192, type=int, help='number of samples for softmax loss')
    training.add_argument('--no_cuda', default=cuda.is_available(), dest='cuda', action='store_false', help='disable CUDA training')
    # elmo options
    elmo = parser.add_argument_group('elmo')
    elmo.add_argument('-ed', '--embedding_dim', default=128, type=int, help='dimension of the word embeddings')
    elmo.add_argument('-cs', '--cell_size', default=512, type=int, help='dimension of the lstm memory cell')
    elmo.add_argument('-l', '--layers', default=2, type=int, help='number of lstm layers')
    elmo.add_argument('-rd', '--recurrent_dropout', default=0.1, type=float, help='recurrent dropout for each lstm')
    elmo.add_argument('-fd', '--dropout', default=0.1, type=float, help='final dropout for elmo representations')
    elmo.add_argument('-sl', '--max_sequence_length', default=400, type=int, help='maximum sequence length (clipped above)')
    # distribution options
    distribution = parser.add_argument_group('distribution')
    distribution.add_argument('-r', '--role', default='scheduler', type=str, help='scheduler or server')
    distribution.add_argument('-ru', '--root_uri', default='127.0.0.1', type=str, help='adress of the scheduler node')
    distribution.add_argument('-rp', '--root_port', default='9091', type=str, help='port of the scheduler node')
    distribution.add_argument('-ps', '--servers', default=1, type=int, help='number of local serve instances to create')
    distribution.add_argument('-ws', '--world_size', default=0, type=int, help='total number of server instances')
    distribution.add_argument('-sd', '--sync_dense', default=1, type=int, help='synchronize dense model parameters every n steps')
    args = parser.parse_args()
    if args.world_size == 0:
        args.world_size = args.servers
    return args


if __name__ == "__main__":
    print(parse_arguments())