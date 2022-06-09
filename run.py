#!/usr/bin/env python3
from datetime import datetime
from functools import partial
import numpy as np
import os
from signal import signal, SIGINT
from sys import exit
from threading import Thread
import torch
from torch.nn import init
from torch import cuda
from torch.multiprocessing import Process, set_start_method
from torch.utils.data import DataLoader

import lapse

from cli import parse_arguments
from data import OneBillionWordIterableDataset
#from elmo import PSElmo
from iterator import PrefetchIterator
from loss import PSSampledSoftmaxLoss
#from optimizer import PSSGD, PSAdagrad
from utils import load_vocab, batch_to_word_ids


def run_worker(worker_id, args, kv):
    train(worker_id, args, kv)
    kv.barrier() # wait for all workers to finish
    kv.finalize()

def loguniform(args):
    log_samples = torch.rand(3000) * np.log(args.num_tokens + 1)
    return torch.exp(log_samples).long() - 1


def train(worker_id, args, kv):
    print(f"Worker {worker_id} training on {args.device}")
    kv.begin_setup()
    if worker_id == 0:
        keys = torch.LongTensor(range(args.num_tokens))
        vals = torch.empty(keys.size()+(2,args.embedding_dim), dtype=torch.float32)
        init.normal_(vals[:,0,:])
        vals[:,1,:] = 1
        kv.set(keys, vals)
    kv.end_setup()
    kv.wait_sync()    
    print(f"Worker {worker_id} sync")

    train_iterator = PrefetchIterator(args.intent_ahead, kv, lambda: prepare_batch(kv, args))
    for i, targets in enumerate(train_iterator):
        keys = targets
        size = keys.size() + (2, args.embedding_dim)
        vals = torch.empty(size, dtype=torch.float32)
        kv.pull(keys, vals)
        if ((vals[:,1,:]) < 0).any():
            zero = torch.nonzero(vals[:,1,:] < 0)
            index = zero[:,0].unique()
            error_keys = keys[index]
            print(f"ALERT: Pulled embedding acc negative:{error_keys} key:range{(torch.min(keys), torch.max(keys))}")
            print(f"key-size:{keys.size()}, buffer size:{vals.size()}")
            for x in error_keys:
                print(x, torch.sum(keys == x))
                print(vals[keys == x])
        
        kv.advance_clock()
        print('[%6d]' % (i))
        if i > 10000:
            break;



def prepare_batch(kv, args):
    target_time = kv.current_clock() + args.intent_ahead

    word_ids = loguniform(args)
    word_ids = word_ids[word_ids > 0]
    targets = torch.cat((word_ids, word_ids))
    targets -= 1

    kv.intent(word_ids.flatten(), target_time)

    return targets

def init_scheduler(dummy, args):
    os.environ['DMLC_NUM_SERVER'] = str(args.world_size)
    os.environ['DMLC_ROLE'] = 'scheduler'
    os.environ['DMLC_PS_ROOT_URI'] = args.root_uri
    os.environ['DMLC_PS_ROOT_PORT'] = args.root_port
    print("running scheduler")
    lapse.scheduler(args.num_keys, args.workers_per_node)


def init_node(local_rank, lens, args):
    """Start up a Lapse node (server + multiple worker threads)"""
    os.environ['DMLC_NUM_SERVER'] = str(args.world_size)
    os.environ['DMLC_ROLE'] = 'server'
    os.environ['DMLC_PS_ROOT_URI'] = args.root_uri
    os.environ['DMLC_PS_ROOT_PORT'] = args.root_port

    print("starting server...")
    lapse.setup(args.num_keys, args.workers_per_node)
    server = lapse.Server(lens)
    rank = server.my_rank()
    print(f"Started server with rank {rank}.")

    # make sure all servers are set up
    server.barrier()

    # setup sampling
    #sample_min = len(PSElmo.lens(args.num_tokens, args.embedding_dim, args.cell_size, args.layers))
    #sample_max = sample_min + args.num_tokens
    #server.enable_sampling_support(args.sampling_scheme, args.sample_replacement, "log-uniform", sample_min, sample_max)

    threads = []
    for w in range(args.workers_per_node):
        worker_id = rank * args.workers_per_node + w

        # assign training device to worker
        if args.cuda:
            local_worker_id = local_rank * args.workers_per_node + w
            if args.device_ids:
                device_id = args.device_ids[local_worker_id]
            else:
                device_id = local_worker_id % cuda.device_count()
            args.device = torch.device("cuda:" + str(device_id))
        else:
            args.device = torch.device("cpu")

        # run worker
        t = Thread(target=run_worker, args=(worker_id, args, lapse.Worker(w, server)))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # shutdown lapse node
    server.shutdown()


def kill_processes(signal_received, frame):
    """Kills all started lapse processes"""
    print('\nSIGINT or CTRL-C detected. Shutting down all processes and exiting..')
    for p in processes:
        p.kill()
    exit(0)

processes = []
if __name__ == "__main__":
    # run cli
    args = parse_arguments()

    # read environment variables when running with tracker
    if args.tracker:
        lapse_env = {'DMLC_NUM_SERVER', 'DMLC_ROLE', 'DMLC_PS_ROOT_URI', 'DMLC_PS_ROOT_PORT'}
        assert os.environ.keys() >= lapse_env, f'Missing Lapse environment variables. Check {lapse_env} are set.'
        args.role = os.environ['DMLC_ROLE']
        args.root_uri = os.environ['DMLC_PS_ROOT_URI']
        args.root_port = os.environ['DMLC_PS_ROOT_PORT']
        args.world_size = int(os.environ['DMLC_NUM_SERVER'])
        if args.role == 'scheduler':
            args.nodes = 0

    print(args)

    # load vocab
    args.vocab2id = load_vocab(args.vocab)
    args.num_tokens = len(args.vocab2id)
    print(f"Loaded vocabulary of {args.num_tokens} tokens.")

    # calculate parameter lens
    #lens_elmo = PSElmo.lens(args.num_tokens, args.embedding_dim, args.cell_size, args.layers)
    #lens_classifier = PSSampledSoftmaxLoss.lens(args.num_tokens, args.embedding_dim)
    #lens = torch.cat((lens_elmo,lens_classifier,torch.ones(1)))
    lens = torch.ones(args.num_tokens) * 2 * args.embedding_dim
    args.num_keys = len(lens)

    # estimate num_tries for loss calculation if necessary
    if not args.num_tries and not args.sample_replacement:
        print("Estimating num_tries...")
        args.num_tries = PSSampledSoftmaxLoss.estimate_num_tries(args.num_tokens, args.samples)
        print(f"AVG num_tries={args.num_tries}")

    # catch interrupt (to shut down lapse processes)
    signal(SIGINT, kill_processes)

    # "spawn" required for cuda training
    set_start_method('spawn')

    # launch lapse scheduler
    if args.role == 'scheduler':
        p = Process(target=init_scheduler, args=(0, args))
        p.start()
        processes.append(p)

    # launch lapse processes
    for local_rank in range(args.nodes):
        p = Process(target=init_node, args=(local_rank, lens, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
