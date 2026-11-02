import torch
import torch.nn.parameter
from collections.abc import Iterable
import math


def gradient_clipping(
    params: Iterable[torch.nn.Parameter],
    max_l2_norm: float
):
    l2_sum = 0.0
    for param in params:
        if param.grad is None:
            continue
        l2_sum += torch.sum(param.grad ** 2)
    l2_norm = torch.sqrt(l2_sum)
    
    if l2_norm > max_l2_norm:
        scale = max_l2_norm / l2_norm
        for param in params:
            if param.grad is None:
                continue
            param.grad.mul_(scale)

def get_lr_cosine_schedue(t, lr_max, lr_min, T_w, T_c) -> float:
    if t < T_w:
        return lr_max * t / T_w
    elif T_w <= t and t<= T_c:
        return lr_min + 0.5 * (1 + math.cos( (t-T_w) / (T_c - T_w) * math.pi)) * (lr_max - lr_min)
    else:
        return lr_min
        
class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        betas: tuple[float, float],
        eps: float = 1e-5,
        weight_decay: float = 0.01,
    ):
        defaults = dict(
            lr=lr, 
            beta1=betas[0], 
            beta2=betas[1], 
            eps=eps, 
            weight_decay=weight_decay, 
            step=0)

        super().__init__(params, defaults)

        for group in self.param_groups:
            for param in group["params"]:
                self.state[param]["m"] = torch.zeros_like(param)
                self.state[param]["v"] = torch.zeros_like(param)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1 = group["beta1"]
            b2 = group["beta2"]
            w_d = group["weight_decay"]
            lr = group["lr"]
            eps = group["eps"]
            group["step"] += 1
            t = group["step"]
            debiaser = (1 - b2**t)**0.5 / (1 - b1**t)

            for param in group["params"]:
                if param.grad is None:
                    continue

                m, v = self.state[param]["m"], self.state[param]["v"]

                # apply weight decay
                param.mul_(1 - lr * w_d)

                # apply graident
                m.mul_(b1).add_(param.grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(param.grad, param.grad, value=1 - b2)
                param.addcdiv_(m, torch.sqrt(v).add_(eps), value = -debiaser * lr)

            
