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
        self._timestamps = []
        self._embeddings = None
        self._accumulator = None
        self._initEmbeddings(pad_zero)

    def _initEmbeddings(self, pad_zero: bool):
        keys = torch.LongTensor(range(self.num_embeddings)) + self.key_offset
        values = torch.empty((self.num_embeddings, self.embedding_dim), dtype=torch.float32)
        init.normal_(values)
        if pad_zero:
            values[0,:] = 0 # 0 embedding
        self.kv.set(keys, values)
        self.kv.set(keys+self.num_embeddings, torch.full(values.size(), self.opt.initial_accumulator_value, dtype=torch.float32))

    def pull_async(self, keys: torch.Tensor):
        keys_embeddings = torch.flatten(keys + self.key_offset)
        keys_accumulator = keys_embeddings + self.num_embeddings
        size = keys.size() + (self.embedding_dim,)
        self._embeddings = torch.empty(size, dtype=torch.float32)
        self._accumulator = torch.empty(size, dtype=torch.float32)
        self._timestamps.append(self.kv.pull(keys_embeddings, self._embeddings))
        self._timestamps.append(self.kv.pull(keys_accumulator, self._accumulator))

    def forward(self, keys: torch.Tensor, device=None) -> torch.Tensor:
        if not self._timestamps:
            self.pull_async(keys)
        while self._timestamps:
            self.kv.wait(self._timestamps.pop())

        embeddings = self._embeddings.to(device=device)
        embeddings.requires_grad_()
        if self.opt:
            embeddings.register_hook(self.grad_hook(keys, self._accumulator, self.opt))
        return embeddings
        
    def grad_hook(self, keys: torch.Tensor, accumulators:torch.Tensor, optimizer: PSOptimizer) -> torch.Tensor:
        keys += self.key_offset
        def hook(grad: torch.Tensor) -> torch.Tensor:
            update_embeddings, update_accumulators = optimizer.update(grad.cpu(), accumulators)
            self.kv.push(keys.flatten(), update_embeddings)
            self.kv.push(keys.flatten()+self.num_embeddings, update_accumulators)
            return grad
        return hook

    def __repr__(self):
       return f"PSEmbedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, range=[{self.key_offset}-{self.key_offset+self.num_embeddings-1}])"