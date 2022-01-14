#!/usr/bin/env python3
from functools import partial
import numpy as np
import os
from signal import signal, SIGINT
from sys import exit
from threading import Thread
import torch
from torch import cuda
from torch.multiprocessing import Process, set_start_method
from torch.utils.data import DataLoader

import lapse

from cli import parse_arguments
from data import OneBillionWordIterableDataset
from elmo import PSElmo
from iterator import PrefetchIterator
from loss import PSSampledSoftmaxLoss
from optimizer import PSSGD, PSAdagrad
from utils import load_vocab, batch_to_word_ids


def run_worker(worker_id, rank, device, vocab2id, args, kv):
    train(worker_id, rank, device, vocab2id, args, kv)
    kv.barrier() # wait for all workers to finish


def train(worker_id, rank, device, vocab2id, args, kv):
    print(f"Worker {worker_id} training on {device}")
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
    elmo.to(device)
    classifier.to(device)

    for epoch in range(args.epochs):
        # set up training data
        collate_fn = partial(collate, kv, worker_id, vocab2id, args)
        train_dataset = OneBillionWordIterableDataset(args.dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size * args.world_size * args.workers_per_node, collate_fn=collate_fn)
        train_iterator = PrefetchIterator(args.localize_ahead, train_loader)
        for i, word_ids in enumerate(train_iterator):
            elmo.pullDenseParameters()
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
            #elmo.pushUpdates()
            print('[%6d] loss: %.3f' % (i, loss.item()))

        kv.barrier() # synchronize workers
        if args.testset:
            loss_key = torch.tensor([args.num_parameters-1])
            loss_val = torch.zeros((1), dtype=torch.float32)
            kv.set(loss_key, loss_val)
            elmo.pullDenseParameters()
            elmo.eval()
            classifier.eval()
            acc_loss = 0
            num_loss = 0
            with torch.no_grad():
                test_dataset = OneBillionWordIterableDataset(args.testset)
                test_loader = DataLoader(test_dataset, batch_size=1 * args.world_size * args.workers_per_node, collate_fn=collate_fun)
                test_iterator = PrefetchIterator(args.num_batches, test_loader)
                for i, batch in enumerate(test_iterator):
                    word_ids = batch_to_word_ids(batch[worker_id::args.world_size * args.workers_per_node], vocab2id, args.max_sequence_length)
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
                    acc_loss = acc_loss + loss
                    num_loss = num_loss + 1
                    print('[%6d] loss: %.3f' % (i, loss.item()))

            elmo.train()
            classifier.train()
            kv.barrier()
            loss_part = acc_loss / (num_loss * args.world_size * args.workers_per_node)
            kv.push(loss_key, loss_part.cpu())
            kv.barrier() # synchronize workers
            kv.pull(loss_key, loss_val)
            print('avg loss: %.3f' % (loss_val).item())

def collate(kv, worker_id, vocab2id, args, batch):
    worker_split = batch[worker_id::args.world_size * args.workers_per_node]
    word_ids = batch_to_word_ids(worker_split, vocab2id, args.max_sequence_length)
    kv.localize(word_ids.flatten())
    return word_ids

def init_scheduler(dummy, args):
    os.environ['DMLC_NUM_SERVER'] = str(args.world_size)
    os.environ['DMLC_ROLE'] = 'scheduler'
    os.environ['DMLC_PS_ROOT_URI'] = args.root_uri
    os.environ['DMLC_PS_ROOT_PORT'] = args.root_port

    lapse.scheduler(args.num_keys, args.workers_per_node)


def init_node(local_rank, lens, vocab2id, args):
    """Start up a Lapse node (server + multiple worker threads)"""
    os.environ['DMLC_NUM_SERVER'] = str(args.world_size)
    os.environ['DMLC_ROLE'] = 'server'
    os.environ['DMLC_PS_ROOT_URI'] = args.root_uri
    os.environ['DMLC_PS_ROOT_PORT'] = args.root_port
    
    lapse.setup(args.num_keys, args.workers_per_node)
    if args.hotspots < 0:
        s = lapse.Server(lens)
    else:
        classifier_offset = len(PSElmo.lens(args.num_tokens, args.embedding_dim, args.cell_size, args.layers))
        hotspots_elmo = PSElmo.hotspots(args.hotspots, args.num_tokens, args.layers)
        hotspots_classifier = PSSampledSoftmaxLoss.hotspots(args.hotspots, args.num_tokens) + classifier_offset
        hotspots = torch.cat((hotspots_elmo, hotspots_classifier))
        s = lapse.Server(lens, hotspots)

    try:
        rank = s.my_rank()
    except:
        rank = local_rank
        print("Failed to fetch the global rank, using local rank instead.")

    print(f"Started server with rank {rank}.")

    threads = []
    for w in range(args.workers_per_node):
        worker_id = rank * args.workers_per_node + w
        kv = lapse.Worker(0, w+1, s)

        # assign training device to worker
        if args.cuda:
            local_worker_id = local_rank * args.workers_per_node + w
            if args.device_ids:
                device_id = args.device_ids[local_worker_id]
            else:
                device_id = local_worker_id % cuda.device_count()
            device = torch.device("cuda:" + str(device_id))
        else:
            device = torch.device("cpu")

        # run worker
        t = Thread(target=run_worker, args=(worker_id, rank, device, vocab2id, args, kv))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # shutdown lapse node
    s.shutdown()


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

    # load vocab
    vocab2id = load_vocab(args.vocab)
    args.num_tokens = len(vocab2id)

    # calculate parameter lens
    lens_elmo = PSElmo.lens(args.num_tokens, args.embedding_dim, args.cell_size, args.layers)
    lens_classifier = PSSampledSoftmaxLoss.lens(args.num_tokens, args.embedding_dim)
    lens = torch.cat((lens_elmo,lens_classifier,torch.ones(1)))
    args.num_keys = len(lens)

    print(args)

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
        p = Process(target=init_node, args=(local_rank, lens, vocab2id, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
