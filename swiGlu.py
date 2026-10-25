import torch
import torch.nn.functional as F
from linear import Linear

def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * F.sigmoid(x)
        

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_fnn, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_fnn = d_fnn
        self.w1 = Linear(d_model, d_fnn, device=device, dtype=dtype)
        self.w2 = Linear(d_fnn, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_fnn, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2((SiLU(self.w1(x)) * self.w3(x)))

        