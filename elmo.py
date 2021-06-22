from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.modules.scalar_mix import ScalarMix
from embedding import PSEmbedding
from optimizer import PSOptimizer
import torch
from torch.nn import Dropout
from typing import List
import lapse


class PSElmo(torch.nn.Module):

    def _lens_lstm(embedding_dim, lstm_cell_size, num_layers):
        layer = [4*lstm_cell_size*embedding_dim, 4*lstm_cell_size*embedding_dim, 4*lstm_cell_size*embedding_dim, 4*lstm_cell_size*embedding_dim, 4*lstm_cell_size, 4*lstm_cell_size, embedding_dim*lstm_cell_size, embedding_dim*lstm_cell_size]
        return torch.tensor(layer*2*num_layers)

    def _lens_scalar_mix(num_layers):
        return torch.tensor([1]*(num_layers+1+1)*2)

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
        estimate_parameters: bool = False,
    ) -> None:
        super().__init__()
        self.kv = kv
        self.opt = opt
        self.word_embedding = PSEmbedding(
            kv=kv,
            key_offset=key_offset,
            num_embeddings=num_tokens+1, 
            embedding_dim=embedding_dim,
            opt=opt,
        )
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
        self.scalar_mix = ScalarMix(
            mixture_size=num_layers + 1,
            do_layer_norm=scalar_mix_do_layer_norm,
            initial_scalar_parameters=scalar_mix_parameters,
            trainable=scalar_mix_parameters is None,
        )
        self.dropout = Dropout(p=dropout)
        self.estimate_parameters = estimate_parameters
        self._lstm_offset = key_offset + len(PSEmbedding.lens(num_tokens+1, embedding_dim))
        self._initParameters()

    def _add_buffer(self, name, tensor):
        self._buffers[name] = tensor
        self._non_persistent_buffers_set.add(name)

    def _initParameters(self):
        for i, (name, param) in enumerate(self.named_parameters()):
            key = torch.tensor([2*i+self._lstm_offset])
            self.kv.set(key, param)
            if self.opt:
                accumulator = torch.full(param.size(), self.opt.initial_accumulator_value)
                self.kv.set(key+1, accumulator)
                self._add_buffer(name, accumulator)
                param.register_hook(self.grad_hook(key, name, self.opt))

    def grad_hook(self, key: torch.Tensor, name, optimizer: PSOptimizer) -> torch.Tensor:
        def hook(grad: torch.Tensor) -> torch.Tensor:
            accumulator = dict(self.named_buffers())[name]
            update_parameter, update_accumulator = optimizer.update(grad, accumulator)
            self.kv.push(key, update_parameter.cpu())
            self.kv.push(key+1, update_accumulator.cpu())
            if self.estimate_parameters:
                with torch.no_grad():
                    accumulator += update_accumulator
                    param = dict(self.named_parameters())[name]
                    param += update_parameter
            return grad
        return hook

    def pullDenseParameters(self):
        with torch.no_grad():
            device = self.scalar_mix.gamma.device
            self.cpu()
            timestamps = []
            accumulators = dict(self.named_buffers())
            for i, (name, param) in enumerate(self.named_parameters()):
                key = torch.tensor([2*i+self._lstm_offset])
                timestamps.append(self.kv.pull(key, param))
                key += 1
                timestamps.append(self.kv.pull(key, accumulators[name]))
            for ts in timestamps:
                self.kv.wait(ts)
            self.to(device)

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
