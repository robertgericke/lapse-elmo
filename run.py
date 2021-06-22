#!/usr/bin/env python3
import os
import torch
from torch import cuda
import torch.distributed as dist
import numpy as np
from optimizer import PSSGD, PSAdagrad
import multiprocessing as mp
import lapse
from signal import signal, SIGINT
from sys import exit
from embedding import PSEmbedding
from loss import PSSampledSoftmaxLoss
from elmo import PSElmo
from utils import load_vocab, batch_to_word_ids
from data import OneBillionWordIterableDataset
from torch.utils.data import DataLoader
from cli import parse_arguments

num_workers_per_server = 1
async_ops = True

localip = '127.0.0.1'
port = '9091'


def train(worker_id, rank, vocab2id, args, kv):
    optimizer = PSAdagrad(
        lr = 0.2,
        initial_accumulator_value=1.0,
        eps = 0,
    )
    elmo = PSElmo(
        kv=kv,
        key_offset=0,
        num_tokens=args.num_tokens,
        embedding_dim=args.embedding_dim,
        num_layers=args.layers,
        lstm_cell_size=args.cell_size,
        lstm_recurrent_dropout=args.recurrent_dropout,
        dropout=args.dropout,
        opt=optimizer,
        estimate_parameters=args.sync_dense>1,
    )
    classifier = PSSampledSoftmaxLoss(
        kv=kv, 
        key_offset=len(PSElmo.lens(args.num_tokens, args.embedding_dim, args.cell_size, args.layers)),
        num_embeddings=args.num_tokens,
        embedding_dim=args.embedding_dim,
        num_samples=args.samples,
        opt=optimizer,
    )
    
    # move model to device
    if args.cuda and cuda.is_available():
        device_id = rank % cuda.device_count()
        device = torch.device("cuda:" + str(device_id))
        elmo.to(device)
        classifier.to(device)
        print(device)

    # set up training
    dataset = OneBillionWordIterableDataset(args.dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size * args.world_size, collate_fn=list)
    for epoch in range(args.epochs):
        for i, batch in enumerate(loader):
            if i % args.sync_dense == 0:
                elmo.pullDenseParameters()
            word_ids = batch_to_word_ids(batch[rank::args.world_size], vocab2id)
            elmo_representation, word_mask = elmo(word_ids)
            mask = word_mask.clone()
            mask[:, 0] = False
            mask_rolled = mask.roll(-1, 1)

            targets_forward = word_ids[mask]
            targets_backward = word_ids[mask_rolled]
            targets = torch.cat((targets_forward, targets_backward))
            targets -= 1 # offset 1-based token ids to 0-based sampling ids
            context_forward = elmo_representation[:, :, :args.embedding_dim][mask_rolled]
            context_backward = elmo_representation[:, :, args.embedding_dim:][mask]
            context = torch.cat((context_forward, context_backward))

            loss = classifier(context, targets) / targets.size(0)
            loss.backward()
            print('[%6d] loss: %.3f' % (i, loss.item()))
        kv.barrier(); # synchronize workers


def init_scheduler(dummy, args):
    os.environ['DMLC_NUM_WORKER'] = '0'
    os.environ['DMLC_NUM_SERVER'] = str(args.servers)
    os.environ['DMLC_ROLE'] = 'scheduler'
    os.environ['DMLC_PS_ROOT_URI'] = localip
    os.environ['DMLC_PS_ROOT_PORT'] = port
    lapse.scheduler(args.num_parameters, num_workers_per_server)


def init_server(rank, lens, vocab2id, args, fn):
    os.environ['DMLC_NUM_WORKER'] = '0'
    os.environ['DMLC_NUM_SERVER'] = str(args.servers)
    os.environ['DMLC_ROLE'] = 'server'
    os.environ['DMLC_PS_ROOT_URI'] = localip
    os.environ['DMLC_PS_ROOT_PORT'] = port
    
    lapse.setup(len(lens), num_workers_per_server)
    s = lapse.Server(lens)

    for w in range(num_workers_per_server):
        worker_id = rank * num_workers_per_server + w
        kv = lapse.Worker(0, w+1, s)
        fn(worker_id, rank, vocab2id, args, kv)
        del kv

    # shutdown server
    s.shutdown()


def kill_processes(signal_received, frame):
    """Kills all started lapse processes"""
    print('\nSIGINT or CTRL-C detected. Shutting down all processes and exiting..')
    for p in processes:
        p.kill()
    exit(0)

processes = []
if __name__ == "__main__":
    args = parse_arguments()

    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    vocab2id = load_vocab(args.vocab)
    args.num_tokens = len(vocab2id)

    lens_elmo = PSElmo.lens(args.num_tokens, args.embedding_dim, args.cell_size, args.layers)
    lens_classifier = PSSampledSoftmaxLoss.lens(args.num_tokens, args.embedding_dim)
    lens = torch.cat((lens_elmo,lens_classifier))
    args.num_parameters = len(lens)

    print(args)

    # catch interrupt (to shut down lapse processes)
    signal(SIGINT, kill_processes)

    # launch lapse scheduler
    p = mp.Process(target=init_scheduler, args=(0, args))
    p.start()
    processes.append(p)

    # launch lapse processes
    for rank in range(args.servers):
        p = mp.Process(target=init_server, args=(rank, lens, vocab2id, args, train))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
