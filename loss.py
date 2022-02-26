from allennlp.modules.sampled_softmax_loss import _choice
from allennlp.nn import util
from embedding import PSEmbedding
from optimizer import PSOptimizer
import torch
import lapse
import numpy as np

# based on allennlp.modules.sampled_softmax_loss
# see: https://github.com/allenai/allennlp/blob/main/allennlp/modules/sampled_softmax_loss.py

class PSSampledSoftmaxLoss(torch.nn.Module):
    def lens(num_embeddings, embedding_dim):
        return PSEmbedding.lens(num_embeddings, embedding_dim+1)

    def __init__(
        self,
        kv: lapse.Worker,
        key_offset: int = 0,
        num_embeddings: int = 1024,
        embedding_dim: int = 512,
        num_samples: int = 128,
        opt: PSOptimizer = None,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self._num_words = num_embeddings
        self._log_num_words_p1 = np.log(num_embeddings + 1)
        self.embedding = PSEmbedding(
            kv=kv, 
            key_offset=key_offset,
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim+1, 
            opt=opt,
        )

    def intent(self, ids, start, stop=0):
        self.embedding.intent(ids, start, stop)

    def pull_async(self, ids: torch.Tensor):
        self.embedding.pull_async(ids)

    def forward(self, embeddings: torch.Tensor, target_ids: torch.Tensor, sample_ids: torch.Tensor = None, samples: torch.Tensor = None, num_tries=None, unique=False) -> torch.Tensor:
        if embeddings.shape[0] == 0: # empty batch
            return torch.tensor(0.0, device=embeddings.device)

        if not self.training:
            return self._forward_eval(embeddings, target_ids, samples)
        else:
            return self._forward_train(embeddings, target_ids, sample_ids, samples, num_tries, unique)

    def _forward_eval(self, embeddings: torch.Tensor, target_ids: torch.Tensor, parameters) -> torch.Tensor:
        w = parameters[:,1:]
        b = parameters[:,:1].flatten()

        log_softmax = torch.nn.functional.log_softmax(torch.matmul(embeddings, w.t()) + b, dim=-1)

        return torch.nn.functional.nll_loss(log_softmax, target_ids.to(embeddings.device), reduction="sum")

    def _forward_train(self, embeddings: torch.Tensor, target_ids: torch.Tensor, sample_ids: torch.Tensor, samples: torch.Tensor, num_tries=None, unique=False) -> torch.Tensor:
        target_expected_count = self.expected_count(target_ids, num_tries=None, unique=False)
        sampled_expected_count = self.expected_count(sample_ids, num_tries=None, unique=False)

        # Get the softmax weights (so we can compute logits)
        true_e = self.embedding(target_ids, embeddings.device)
        true_w = true_e[:,1:]
        true_b = true_e[:,:1].flatten()
        sampled_e = samples.to(embeddings.device)
        sampled_w = sampled_e[:,1:]
        sampled_b = sampled_e[:,:1].flatten()

        # compute the logits and remove log expected counts
        # [batch_size, ]
        true_logits = (
            (true_w * embeddings).sum(dim=1)
            + true_b
            - torch.log(
                target_expected_count.to(embeddings.device) + util.tiny_value_of_dtype(target_expected_count.dtype)
            )
        )

        # [batch_size, n_samples]
        sampled_logits = (
            torch.matmul(embeddings, sampled_w.t())
            + sampled_b
            - torch.log(
                sampled_expected_count.to(embeddings.device) + util.tiny_value_of_dtype(sampled_expected_count.dtype)
            )
        )
        
        # remove true labels -- we will take
        # softmax, so set the sampled logits of true values to a large
        # negative number
        # [batch_size, n_samples]
        true_in_sample_mask = sample_ids == target_ids.unsqueeze(1)
        masked_sampled_logits = sampled_logits.masked_fill(true_in_sample_mask.to(embeddings.device), -10000.0)
        # now concat the true logits as index 0
        # [batch_size, n_samples + 1]
        logits = torch.cat([true_logits.unsqueeze(1), masked_sampled_logits], dim=1)

        # finally take log_softmax
        log_softmax = torch.nn.functional.log_softmax(logits, dim=1)
        # true log likelihood is index 0, loss = -1.0 * sum over batch
        # the likelihood loss can become very large if the corresponding
        # true logit is very small, so we apply a per-target cap here
        # so that a single logit for a very rare word won't dominate the batch.
        nll_loss = -1.0 * log_softmax[:, 0].sum()
        return nll_loss

    def sample(self, unique=False):
        if unique: # no sample will appear more than once
            np_sampled_ids, num_tries = choice_func(self._num_words, self.num_samples)
            return torch.from_numpy(np_sampled_ids), num_tries
        else:
            log_samples = np.random.rand(self.num_samples) * np.log(self._num_words + 1)
            samples = np.exp(log_samples).astype("int64") - 1
            return torch.from_numpy(np.clip(samples, a_min=0, a_max=self._num_words - 1))

    def expected_count(self, ids, num_tries=None, unique=False):
        # see: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/range_sampler.h
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/range_sampler.cc

        # algorithm: keep track of number of tries when doing sampling, then expected count is:
        # -expm1(num_tries * log1p(-p)) = (1 - (1-p)^num_tries) where p is self._probs[id]

        # Compute expected count = (1 - (1-p)^num_tries) = -expm1(num_tries * log1p(-p))
        # P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)
        probs = torch.log((ids.float() + 2.0) / (ids.float() + 1.0)) / self._log_num_words_p1
        if unique:
            expected_count = -1.0 * (torch.exp(num_tries * torch.log1p(-probs)) - 1.0)
        else:
            expected_count = probs * self.num_samples
        return expected_count
