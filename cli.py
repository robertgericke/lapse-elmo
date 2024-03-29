from argparse import ArgumentParser
from torch import cuda

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str, help='directory of training data')
    parser.add_argument('vocab', type=str, help='vocabulary file')
    parser.add_argument('-ts', '--testset', type=str, help='directory of test data')
    # cuda options
    cudaoptions = parser.add_argument_group('cuda')
    cudaoptions.add_argument('--no_cuda', default=cuda.is_available(), dest='cuda', action='store_false', help='disable CUDA training')
    cudaoptions.add_argument('-di', '--device_ids', nargs='+', type=int, help='IDs of cuda devices for each worker')
    # distribution options
    distribution = parser.add_argument_group('distribution')
    distribution.add_argument('-r', '--role', type=str, help='scheduler or server')
    distribution.add_argument('-ru', '--root_uri', default='127.0.0.1', type=str, help='adress of the scheduler node')
    distribution.add_argument('-rp', '--root_port', default='9091', type=str, help='port of the scheduler node')
    distribution.add_argument('-nn', '--nodes', default=1, type=int, help='number of local server nodes to create')
    distribution.add_argument('-ia', '--intent_ahead', default=1, type=int, help='number of training batches to pre load and intent')
    distribution.add_argument('-nw', '--workers_per_node', default=1, type=int, help='number of worker threads per node')
    distribution.add_argument('-ws', '--world_size', type=int, help='total number of server nodes')
    distribution.add_argument('-t', '--tracker', default=False, action='store_true', help='use this option if running with adaps tracker scripts')    
    # elmo options
    elmo = parser.add_argument_group('elmo')
    elmo.add_argument('-ed', '--embedding_dim', default=128, type=int, help='dimension of the word embeddings')
    elmo.add_argument('-cs', '--cell_size', default=512, type=int, help='dimension of the lstm memory cell')
    elmo.add_argument('-l', '--layers', default=2, type=int, help='number of lstm layers')
    elmo.add_argument('-rd', '--recurrent_dropout', default=0.1, type=float, help='recurrent dropout for each lstm')
    elmo.add_argument('-fd', '--dropout', default=0.1, type=float, help='final dropout for elmo representations')
    elmo.add_argument('-sl', '--max_sequence_length', default=400, type=int, help='maximum sequence length (clipped above)')
    # optimizer parameters
    optimizer = parser.add_argument_group('optimizer')
    optimizer.add_argument('-eps', '--epsilon', default=1e-07, type=float, help='adagrad epsilon')
    optimizer.add_argument('-iac', '--initial_accumulator_value', default=1.0, type=float, help='adagrad initial accumulator value')
    optimizer.add_argument('-lr', '--learning_rate', default=0.2, type=float, help='adagrad learning rate')
    # training options
    training = parser.add_argument_group('training')
    training.add_argument('-e', '--epochs', default=2, type=int, help='number of epochs to run')
    training.add_argument('-b', '--batch_size', default=128, type=int, help='number of sentences per batch')
    training.add_argument('-s', '--samples', default=8192, type=int, help='number of samples for softmax loss')
    training.add_argument('-sr', '--sample_replacement', dest='sample_replacement', default=False, action='store_true', help='sample witht replacement')
    training.add_argument('-ss', '--sampling_scheme', default='preloc', type=str, help='server sampling method')
    training.add_argument('-nt', '--num_tries', type=float, help='expected number of tries to get num_samples unique samples')

    args = parser.parse_args()
    if args.world_size is None:
        args.world_size = args.nodes
        args.role = 'scheduler'
    return args


if __name__ == "__main__":
    print(parse_arguments())