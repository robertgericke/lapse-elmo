import torch
from torch.nn import init
from optimizer import PSOptimizer
import adaps
import numpy as np


class PSEmbedding(torch.nn.Module):

    def lens(num_embeddings, embedding_dim, bias=False):
        if bias:
            embedding_dim += 1
        return torch.ones(num_embeddings) * embedding_dim * 2 #twice embedding_dim for optim params

    def __init__(
        self,
        kv: adaps.Worker,
        key_offset: int = 0,
        num_embeddings: int = 1024,
        embedding_dim: int = 512,
        bias: bool = False,
        opt: PSOptimizer = None,
        init: bool = True,
        max_size: int = 2**14
    ) -> None:
        super().__init__()
        self.kv = kv
        self.key_offset = key_offset
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if bias:
            self.embedding_dim += 1
        self.opt = opt
        self._buffer = None
        self.max_size = max_size
        if init:
            self._init_embeddings(bias)

    def _init_embeddings(self, bias):
        for ids in torch.LongTensor(range(self.num_embeddings)).split(self.max_size):
            keys = ids + self.key_offset
            values = torch.empty(keys.size()+(2,self.embedding_dim), dtype=torch.float32)
            init.normal_(PSEmbedding._embeddings(values))
            if bias:
                values /= np.sqrt(self.embedding_dim-1)
                values[:,0,0] = 0
            PSEmbedding._accumulators(values)[:] = self.opt.initial_accumulator_value
            self.kv.set(keys, values)

    def _embeddings(buffer):
        slice_dim = buffer.dim() - 2
        return buffer.select(slice_dim, 0)

    def _accumulators(buffer):
        slice_dim = buffer.dim() - 2
        return buffer.select(slice_dim, 1)

    def intent(self, ids: torch.Tensor, start, stop = 0):
        keys = ids.flatten() + self.key_offset
        self.kv.intent(keys, start, stop)

    def pull(self, ids: torch.Tensor):
        keys = ids.flatten() + self.key_offset
        size = ids.size() + (2, self.embedding_dim)
        self._buffer = torch.empty(size, dtype=torch.float32)
        self.kv.pull(keys, self._buffer)

    def forward(self, ids: torch.Tensor, device=None) -> torch.Tensor:
        if self._buffer is None:
            self.pull(ids)

        embeddings = PSEmbedding._embeddings(self._buffer).to(device=device).requires_grad_()
        if self.opt:
            embeddings.register_hook(self.grad_hook(ids))

        return embeddings

    def grad_hook(self, ids: torch.Tensor) -> torch.Tensor:
        def hook(grad: torch.Tensor) -> torch.Tensor:
            keys = ids.flatten() + self.key_offset
            self.opt.update_in_place(grad.cpu(), PSEmbedding._embeddings(self._buffer), PSEmbedding._accumulators(self._buffer))
            self.kv.push(keys, self._buffer, True)
            self._buffer = None
            return grad
        return hook

    def __repr__(self):
       return f"PSEmbedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, range=[{self.key_offset}-{self.key_offset+self.num_embeddings-1}])"
