import torch
from torch.nn import init
from optimizer import PSOptimizer
import lapse


class PSEmbedding(torch.nn.Module):

    def lens(num_embeddings, embedding_dim):
        return torch.ones(num_embeddings*2) * embedding_dim

    def __init__(
        self,
        kv: lapse.Worker,
        key_offset: int = 0,
        num_embeddings: int = 1024,
        embedding_dim: int = 512,
        opt: PSOptimizer = None,
        pad_zero: bool = False,
    ) -> None:
        super().__init__()
        self.kv = kv
        self.key_offset = key_offset
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.opt = opt
        self._initEmbeddings(pad_zero)

    def _initEmbeddings(self, pad_zero: bool):
        keys = torch.LongTensor(range(self.num_embeddings)) + self.key_offset
        values = torch.empty((self.num_embeddings, self.embedding_dim), dtype=torch.float32)
        init.normal_(values)
        if pad_zero:
            values[0,:] = 0 # 0 embedding
        ts = self.kv.set(keys, values)
        ts = self.kv.set(keys+self.num_embeddings, torch.full(values.size(), self.opt.initial_accumulator_value, dtype=torch.float32))

    def forward(self, keys: torch.Tensor, device=None) -> torch.Tensor:
        keys = keys + self.key_offset
        size = keys.size() + (self.embedding_dim,)
        embeddings = torch.empty(size, dtype=torch.float32)
        accumulators = torch.empty(size, dtype=torch.float32)
        ts1 = self.kv.pull(keys.flatten(), embeddings)
        ts2 = self.kv.pull(keys.flatten()+self.num_embeddings, accumulators)
        self.kv.wait(ts1)
        self.kv.wait(ts2)
        embeddings = embeddings.to(device=device)
        accumulators = accumulators
        embeddings.requires_grad_()
        if self.opt:
            embeddings.register_hook(self.grad_hook(keys, accumulators, self.opt))
        return embeddings
        
    def grad_hook(self, keys: torch.Tensor, accumulators:torch.Tensor, optimizer: PSOptimizer) -> torch.Tensor:
        def hook(grad: torch.Tensor) -> torch.Tensor:
            update_embeddings, update_accumulators = optimizer.update(grad.cpu(), accumulators)
            self.kv.push(keys.flatten(), update_embeddings)
            self.kv.push(keys.flatten()+self.num_embeddings, update_accumulators)
            return grad
        return hook

    def __repr__(self):
       return f"PSEmbedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, range=[{self.key_offset}-{self.key_offset+self.num_embeddings-1}])"