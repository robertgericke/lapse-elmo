from collections import deque 
import torch
from torch.utils.data import DataLoader
import lapse

class PrefetchIterator:
    def __init__(self, num_batches, dataloader):
        self.num_batches = num_batches
        self.loader = dataloader

    def __iter__(self):
        self.iter = iter(self.loader)
        self.iter_done = False
        self.queue = deque()
        
        while len(self.queue) < self.num_batches:
            self.loadnext()
        
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
