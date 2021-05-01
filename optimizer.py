import torch

class PSOptimizer():
    pass

class PSSGD(PSOptimizer):
    def __init__(self, eta: float = 0.5) -> None:
        self.eta = eta 

    def update(self, gradients: torch.Tensor) -> torch.Tensor:
        return -self.eta * gradients

#class PSAdagrad(PSOptimizer):
#    def __init__(self, eta: float = 0.01, eps: float = 1e-8) -> None:
#        self.eta = eta 
#        self.eps = eps

#    def update(self, delta_w: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
#        v = v.to(delta_w.device)
#        update_v = delta_w * delta_w
#        update_w = -self.eta * delta_w / torch.sqrt(v+update_v+self.eps)
#        return update_w, update_v
        
        
class PSAdagrad(PSOptimizer):
    def __init__(self, lr: float = 0.01, initial_accumulator_value: float = 0.1, eps: float = 1e-10) -> None:
        self.lr = lr 
        self.eps = eps
        self.initial_accumulator_value = initial_accumulator_value

    def update(self, gradient: torch.Tensor, accumulator: torch.Tensor) -> torch.Tensor:
        accumulator = accumulator.to(gradient.device)
        update_accumulator = gradient * gradient
        update_gradient = -self.lr * gradient / (torch.sqrt(accumulator+update_accumulator) + self.eps)
        return update_gradient, update_accumulator