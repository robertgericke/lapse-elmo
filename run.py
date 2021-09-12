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
import threading
from embedding import PSEmbedding
from loss import PSSampledSoftmaxLoss
from elmo import PSElmo
from utils import load_vocab, batch_to_word_ids
from data import OneBillionWordIterableDataset
from torch.utils.data import DataLoader
from cli import parse_arguments


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
    elmo.to(device)
    classifier.to(device)

    for epoch in range(args.epochs):
        # set up training data
        train_dataset = OneBillionWordIterableDataset(args.dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size * args.world_size * args.workers_per_node, collate_fn=list)
        for i, batch in enumerate(train_loader):
            if i % args.sync_dense == 0:
                elmo.pullDenseParameters()
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
            loss.backward()
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
                test_loader = DataLoader(test_dataset, batch_size=1 * args.world_size * args.workers_per_node, collate_fn=list)
                for i, batch in enumerate(test_loader):
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


def init_scheduler(dummy, args):
    os.environ['DMLC_NUM_SERVER'] = str(args.world_size)
    os.environ['DMLC_ROLE'] = 'scheduler'
    os.environ['DMLC_PS_ROOT_URI'] = args.root_uri
    os.environ['DMLC_PS_ROOT_PORT'] = args.root_port
    lapse.scheduler(args.num_parameters, args.workers_per_node)


def init_node(local_rank, lens, vocab2id, args, fn):
    """Start up a Lapse node (server + multiple worker threads)"""
    os.environ['DMLC_NUM_SERVER'] = str(args.world_size)
    os.environ['DMLC_ROLE'] = 'server'
    os.environ['DMLC_PS_ROOT_URI'] = args.root_uri
    os.environ['DMLC_PS_ROOT_PORT'] = args.root_port
    
    lapse.setup(len(lens), args.workers_per_node)
    s = lapse.Server(lens)
    try:
        rank = s.my_rank()
    except:
        rank = local_rank
        print("failed to fetch rank, using local rank instead")

    print(f"started server with rank {rank}")

    threads = []
    for w in range(args.workers_per_node):
        worker_id = rank * args.workers_per_node + w
        kv = lapse.Worker(0, w+1, s)
        device = torch.device("cpu")
        if args.cuda and cuda.is_available():
            local_worker_id = local_rank * args.workers_per_node + w
            if args.device_ids is None:
                device_id = local_worker_id % cuda.device_count()
            else:
                device_id = args.device_ids[local_worker_id]
            device = torch.device("cuda:" + str(device_id))

        t = threading.Thread(target=fn, args=(worker_id, rank, device, vocab2id, args, kv))
        t.start()
        threads.append(t)
        del kv

    for t in threads:
        t.join()

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
    lens = torch.cat((lens_elmo,lens_classifier,torch.ones(1)))
    args.num_parameters = len(lens)

    print(args)

    # catch interrupt (to shut down lapse processes)
    signal(SIGINT, kill_processes)

    if args.role == 'scheduler':
        # launch lapse scheduler
        p = mp.Process(target=init_scheduler, args=(0, args))
        p.start()
        processes.append(p)

    # launch lapse processes
    for local_rank in range(args.nodes):
        p = mp.Process(target=init_node, args=(local_rank, lens, vocab2id, args, train))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
