#!/usr/bin/env python3
from datetime import datetime
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

import adaps

from cli import parse_arguments
from data import OneBillionWordIterableDataset
from elmo import PSElmo
from iterator import PrefetchIterator
from loss import PSSampledSoftmaxLoss
from optimizer import PSSGD, PSAdagrad
from utils import load_vocab, batch_to_word_ids


def run_worker(worker_id, args, kv):
    train(worker_id, args, kv)
    kv.barrier() # wait for all workers to finish
    kv.finalize()


def train(worker_id, args, kv):
    print(f"Worker {worker_id} training on {args.device}")
    kv.begin_setup()
    optimizer = PSAdagrad(
        lr = 0.2,
        initial_accumulator_value=1.0,
        eps = 1e-07,
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
        init=(worker_id==0),
    )
    classifier = PSSampledSoftmaxLoss(
        kv=kv,
        key_offset=len(PSElmo.lens(args.num_tokens, args.embedding_dim, args.cell_size, args.layers)),
        num_embeddings=args.num_tokens,
        embedding_dim=args.embedding_dim,
        num_samples=args.samples,
        opt=optimizer,
        init=(worker_id==0),
    )
    # move model to computing device
    elmo.to(args.device)
    classifier.to(args.device)
    kv.end_setup()
    kv.wait_sync()

    for epoch in range(args.epochs):
        if worker_id == 0:
            print(f"Starting epoch {epoch}")
        # set up training data
        train_dataset = OneBillionWordIterableDataset(args.dataset)
        train_collate = partial(prepare_batch, kv, worker_id, elmo, classifier, True, args)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size * args.world_size * args.workers_per_node, collate_fn=train_collate)
        train_iterator = PrefetchIterator(args.intent_ahead, kv, train_loader)
        print(f"Begin epoch {epoch} at {datetime.now()}")
        for i, (word_ids, mask, mask_rolled, targets, sample_id) in enumerate(train_iterator):
            elmo.pull_dense_and_embeddings(word_ids)
            classifier.pull(targets)
            sample_ids, samples = pull_samples(kv, sample_id, classifier, optimizer, args.device, args)

            elmo_representation, word_mask = elmo(word_ids)
            context_forward = elmo_representation[:, :, :args.embedding_dim][mask_rolled]
            context_backward = elmo_representation[:, :, args.embedding_dim:][mask]
            context = torch.cat((context_forward, context_backward))

            loss = classifier(context, targets, sample_ids, samples, args.num_tries, args.sample_replacement) / targets.size(0)
            loss.backward()
            kv.advance_clock()
            print('[%6d] loss: %.3f' % (i, loss.item()))
        print(f"Finished epoch {epoch} at {datetime.now()}")

        # synchronize replicas
        kv.wait_sync()
        kv.barrier()
        kv.wait_sync()

        # calculate test loss
        if args.testset:
            elmo.pull_dense_parameters()
            elmo.eval()
            classifier.eval()
            with torch.no_grad():
                test_dataset = OneBillionWordIterableDataset(args.testset)
                test_collate = partial(prepare_batch, kv, worker_id, elmo, classifier, False, args)
                test_loader = DataLoader(test_dataset, batch_size=1 * args.world_size * args.workers_per_node, collate_fn=test_collate)
                test_iterator = PrefetchIterator(args.intent_ahead, kv, test_loader)
                all_ids = torch.tensor(range(args.num_tokens))
                all_weights = classifier.embedding(all_ids, args.device)
                acc_loss = 0
                num_loss = 0
                for i, (word_ids, mask, mask_rolled, targets) in enumerate(test_iterator):
                    elmo.word_embedding.pull(word_ids)

                    elmo_representation, word_mask = elmo(word_ids)
                    context_forward = elmo_representation[:, :, :args.embedding_dim][mask_rolled]
                    context_backward = elmo_representation[:, :, args.embedding_dim:][mask]
                    context = torch.cat((context_forward, context_backward))

                    loss = classifier(context, targets, samples=all_weights) / targets.size(0)
                    acc_loss = acc_loss + loss
                    num_loss = num_loss + 1
                    kv.advance_clock()
                    print('[%6d] loss: %.3f' % (i, loss.item()))

                # allreduce average loss
                loss_key = torch.tensor([kv.num_keys-1])
                loss_val = torch.zeros((1), dtype=torch.float32)
                kv.set(loss_key, loss_val)
                kv.wait_sync()
                kv.barrier() # synchronize replicas
                kv.wait_sync()
                kv.push(loss_key, (acc_loss / num_loss).cpu())
                kv.wait_sync()
                kv.barrier() # synchronize replicas
                kv.wait_sync()
                if worker_id == 0:
                    kv.pull(loss_key, loss_val)
                    avg_loss = loss_val / (args.world_size * args.workers_per_node)
                    print('avg test loss: %.3f' % (avg_loss).item())

            elmo.train()
            classifier.train()


def prepare_batch(kv, worker_id, elmo, classifier, sample, args, batch):
    target_time = kv.current_clock() + args.intent_ahead

    worker_split = batch[worker_id::args.world_size * args.workers_per_node]
    word_ids = batch_to_word_ids(worker_split, args.vocab2id, args.max_sequence_length)
    elmo.intent_embeddings(word_ids.flatten(), target_time)

    mask = word_ids > 0
    mask[:, 0] = False
    mask_rolled = mask.roll(-1, 1)
    targets_forward = word_ids[mask]
    targets_backward = word_ids[mask_rolled]
    targets = torch.cat((targets_forward, targets_backward))
    targets -= 1 # offset 1-based token ids to 0-based sampling ids

    if not sample:
        return word_ids, mask, mask_rolled, targets

    classifier.intent(word_ids.flatten()-1, target_time)
    sample_id = kv.prepare_sample(args.samples, target_time)

    return word_ids, mask, mask_rolled, targets, sample_id

def pull_samples(kv, sample_id, classifier, opt, device, args):
    keys = torch.empty((args.samples), dtype=torch.long)
    vals = torch.empty((args.samples, 2, args.embedding_dim+1))
    kv.wait(kv.pull_sample(sample_id, keys, vals))
    ids = keys - classifier.embedding.key_offset
    samples = vals[:,0,:].to(device)
    samples.requires_grad_()
    samples.register_hook(grad_hook(kv, keys, vals, opt))

    return ids, samples

def grad_hook(kv, keys: torch.Tensor, vals: torch.Tensor, optimizer) -> torch.Tensor:
    def hook(grad: torch.Tensor) -> torch.Tensor:
        optimizer.update_in_place(grad.cpu(), vals[:,0,:], vals[:,1,:])
        kv.push(keys, vals, True)
        return grad
    return hook

def init_scheduler(dummy, args):
    os.environ['DMLC_NUM_SERVER'] = str(args.world_size)
    os.environ['DMLC_ROLE'] = 'scheduler'
    os.environ['DMLC_PS_ROOT_URI'] = args.root_uri
    os.environ['DMLC_PS_ROOT_PORT'] = args.root_port
    print("running scheduler")
    adaps.scheduler(args.num_keys, args.workers_per_node)


def init_node(local_rank, lens, args):
    """Start up a Lapse node (server + multiple worker threads)"""
    os.environ['DMLC_NUM_SERVER'] = str(args.world_size)
    os.environ['DMLC_ROLE'] = 'server'
    os.environ['DMLC_PS_ROOT_URI'] = args.root_uri
    os.environ['DMLC_PS_ROOT_PORT'] = args.root_port

    adaps.setup(args.num_keys, args.workers_per_node)
    server = adaps.Server(lens)
    rank = server.my_rank()
    print(f"Started server with rank {rank}.")

    # make sure all servers are set up
    server.barrier()

    # setup sampling
    sample_min = len(PSElmo.lens(args.num_tokens, args.embedding_dim, args.cell_size, args.layers))
    sample_max = sample_min + args.num_tokens
    server.enable_sampling_support(args.sampling_scheme, args.sample_replacement, "log-uniform", sample_min, sample_max)

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
        t = Thread(target=run_worker, args=(worker_id, args, adaps.Worker(w, server)))
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
    lens_elmo = PSElmo.lens(args.num_tokens, args.embedding_dim, args.cell_size, args.layers)
    lens_classifier = PSSampledSoftmaxLoss.lens(args.num_tokens, args.embedding_dim)
    lens = torch.cat((lens_elmo,lens_classifier,torch.ones(1)))
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
