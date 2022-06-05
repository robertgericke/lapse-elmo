from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.modules.scalar_mix import ScalarMix
from embedding import PSEmbedding
from optimizer import PSOptimizer
import torch
from torch.nn import Dropout, Parameter
from typing import List
import lapse
import functools
import sys


def rgetattr(obj, path): # recursive getattr: eg. elmo.scalar_mix.gamma
    return functools.reduce(getattr, path.split('.'), obj)

def rsetattr(obj, path, val): # recursive setattr: eg. elmo.scalar_mix.gamma
    pre, _, post = path.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

class PSElmo(torch.nn.Module):

    def _lens_lstm(embedding_dim, lstm_cell_size, num_layers):
        layer = [4*lstm_cell_size*embedding_dim, 4*lstm_cell_size*embedding_dim, 4*lstm_cell_size, embedding_dim*lstm_cell_size]
        return torch.tensor(layer*2*num_layers)*2

    def _lens_scalar_mix(num_layers):
        return torch.tensor([1]*(num_layers+1+1))*2

    def _lens_embedding(num_tokens, embedding_dim):
        return PSEmbedding.lens(num_tokens+1, embedding_dim)

    def lens(num_tokens, embedding_dim, lstm_cell_size, num_layers):
        lens_embedding = PSElmo._lens_embedding(num_tokens, embedding_dim)
        lens_lstm = PSElmo._lens_lstm(embedding_dim, lstm_cell_size, num_layers)
        lens_scalar_mix = PSElmo._lens_scalar_mix(num_layers)
        return torch.cat((lens_embedding, lens_lstm, lens_scalar_mix))

    def __init__(
        self,
        kv: lapse.Worker,
        num_tokens: int,
        key_offset: int = 0,
        embedding_dim: int = 512,
        num_layers: int = 2,
        lstm_cell_size: int = 4096,
        lstm_requires_grad: bool = True,
        lstm_recurrent_dropout: float = 0.1,
        lstm_state_proj_clip: float = 3.0,
        lstm_memory_cell_clip: float = 3.0,
        scalar_mix_do_layer_norm: bool = False,
        scalar_mix_parameters: List[float] = None,
        dropout: float = 0.1,
        opt: PSOptimizer = None,
        init: bool = True,
    ) -> None:
        super().__init__()
        self.kv = kv
        self.opt = opt
        print("init embedding")
        self.word_embedding = PSEmbedding(
            kv=kv,
            key_offset=key_offset,
            num_embeddings=num_tokens+1, #one more for zero embedding (no word / blank space)
            embedding_dim=embedding_dim,
            opt=opt,
            init=init,
        )
        print("init lstm")
        self.elmo_lstm = ElmoLstm(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            cell_size=lstm_cell_size,
            num_layers=num_layers,
            requires_grad=lstm_requires_grad,
            recurrent_dropout_probability=lstm_recurrent_dropout,
            state_projection_clip_value=lstm_state_proj_clip,
            memory_cell_clip_value=lstm_memory_cell_clip,
        )
        print("init scalar")
        self.scalar_mix = ScalarMix(
            mixture_size=num_layers + 1,
            do_layer_norm=scalar_mix_do_layer_norm,
            initial_scalar_parameters=scalar_mix_parameters,
            trainable=scalar_mix_parameters is None,
        )
        self.dropout = Dropout(p=dropout)
        self._lstm_offset = key_offset + len(PSEmbedding.lens(num_tokens+1, embedding_dim))
        self._param_buffers = {}
        print("intent params")
        self.intent_dense_parameters()
        for i, (name, param) in enumerate(self.named_parameters()):
            key = torch.tensor([i+self._lstm_offset])
            self._param_buffers[name] = torch.empty((2,)+param.size())
            param.register_hook(self.grad_hook(key, name))
        if init:
            print("init params")
            self._initParameters()
        self.isfinite = True


    def _initParameters(self):
        for i, (name, param) in enumerate(self.named_parameters()):
            key = torch.tensor([i+self._lstm_offset])
            self._param_buffers[name][0] = param.clone().detach()
            self._param_buffers[name][1] = self.opt.initial_accumulator_value
            self.kv.set(key, self._param_buffers[name])


    def grad_hook(self, key: torch.Tensor, name) -> torch.Tensor:
        def hook(grad: torch.Tensor) -> torch.Tensor:
            self.opt.update_in_place(grad.cpu(), self._param_buffers[name][0], self._param_buffers[name][1])
            self.kv.push(key, self._param_buffers[name])
            if not self._param_buffers[name].isfinite().all():
                print(f"ALERT: LSTM is not finite in:{key}:{name}")
                print(f"grad is finite:{grad.isfinite().all()}")
                self.isfinite = False
            return grad
        return hook

    def intent_embeddings(self, ids: torch.Tensor, start, stop = 0):
        self.word_embedding.intent(ids, start, stop)

    def intent_dense_parameters(self, start = 0, stop = sys.maxsize):
        num_parameters = sum(1 for i in self.parameters())
        ps_keys = torch.arange(num_parameters) + self._lstm_offset
        self.kv.intent(ps_keys, start, stop)

    def pull_dense_and_embeddings(self, ids: torch.Tensor,):
        self.pull_dense_parameters()
        self.word_embedding.pull(ids)

    def pull_dense_parameters(self):
        for i, (name, param) in enumerate(self.named_parameters()):
            key = torch.tensor([i+self._lstm_offset])
            self.kv.pull(key, self._param_buffers[name])
            newParam = Parameter(self._param_buffers[name][0].to(param.device))
            newParam.register_hook(self.grad_hook(key, name))
            rsetattr(self, name, newParam)

    def forward(self, inputs: torch.Tensor) -> (torch.Tensor, torch.BoolTensor):
        device = self.scalar_mix.gamma.device
        embedded = self.word_embedding(inputs, device)
        mask = (inputs > 0).to(device)
        lstm_out = self.elmo_lstm(embedded, mask)
        layer_outputs = [torch.cat([embedded, embedded], dim=-1) * mask.unsqueeze(-1)]
        for layer_activations in torch.chunk(lstm_out, lstm_out.size(0), dim=0):
            layer_outputs.append(layer_activations.squeeze(0))
        elmo_representation = self.scalar_mix(layer_outputs, mask)
        return self.dropout(elmo_representation), mask
