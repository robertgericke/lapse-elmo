import torch
from torch.nn import init
from optimizer import PSOptimizer
import lapse


class PSEmbedding(torch.nn.Module):

    def lens(num_embeddings, embedding_dim):
        return torch.ones(num_embeddings) * embedding_dim * 2 #twice embedding_dim for optim params

    def __init__(
        self,
        kv: lapse.Worker,
        key_offset: int = 0,
        num_embeddings: int = 1024,
        embedding_dim: int = 512,
        opt: PSOptimizer = None,
        init: bool = True,
        max_size: int = 2**14
    ) -> None:
        super().__init__()
        self.kv = kv
        self.key_offset = key_offset
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.opt = opt
        self._buffer = None
        self.max_size = max_size
        self.isfinite = True
        if init:
            self._init_embeddings()

    def _init_embeddings(self, ):
        for ids in torch.LongTensor(range(self.num_embeddings)).split(self.max_size):
            print(ids.data[0])
            keys = ids + self.key_offset
            values = torch.empty(keys.size()+(self.embedding_dim*2,), dtype=torch.float32)
            init.normal_(values[:,:self.embedding_dim])
            values[:,self.embedding_dim:] = self.opt.initial_accumulator_value
            self.kv.set(keys, values)

    def _embeddings(self):
        slice_dim = self._buffer.dim() - 2
        return self._buffer.select(slice_dim, 0)

    def _accumulators(self):
        slice_dim = self._buffer.dim() - 2
        return self._buffer.select(slice_dim, 1)

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

        embeddings = self._embeddings().to(device=device).requires_grad_()
        if self.opt:
            embeddings.register_hook(self.grad_hook(ids))

        return embeddings
        
    def grad_hook(self, ids: torch.Tensor) -> torch.Tensor:
        def hook(grad: torch.Tensor) -> torch.Tensor:
            keys = ids.flatten() + self.key_offset
            buffer = self._buffer.detach().clone()
            self.opt.update_in_place(grad.cpu(), self._embeddings(), self._accumulators())
            self.kv.push(keys, self._buffer, True)
            if not self._buffer.isfinite().all():
                print(f"ALERT: Embedding is not finite in:{(torch.min(keys),torch.max(keys))}")
                print(f"grad is finite:{grad.isfinite().all()}")
                print(f"buffer if finite:{buffer.isfinite().all()}")
                torch.save(grad.cpu(), 'grad.pt')
                torch.save(buffer, 'buffer.pt')
                self.isfinite = False
            self._buffer = None
            return grad
        return hook

    def __repr__(self):
       return f"PSEmbedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, range=[{self.key_offset}-{self.key_offset+self.num_embeddings-1}])"
