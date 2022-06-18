from collections import deque
import torch
from torch.utils.data import DataLoader
import adaps


class PrefetchIterator:
    """ Wrapper for a DataLoader - pre loads num_batches many batches
    (intent signaling may occure in the collate function of the DataLoader)
    """
    def __init__(self, num_batches, kv, dataloader):
        self.num_batches = num_batches
        self.loader = dataloader
        self.kv = kv

    def __iter__(self):
        self.iter = iter(self.loader)
        self.iter_done = False
        self.queue = deque()

        # fill queue with initial num_batches
        for _ in range(self.num_batches):
            self.loadnext()
            self.kv.advance_clock()

        return self

    def loadnext(self):
        if not self.iter_done:
            try:
                self.queue.append(next(self.iter))
            except StopIteration:
                self.iter_done = True

    def __next__(self):
        self.loadnext()

        if not self.queue:
            raise StopIteration

        return self.queue.popleft()
