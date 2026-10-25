import torch

class Linear (torch.nn.Module):
    def __init__(self, in_features, out_features, weights=None, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if weights is not None:
            self.weight = torch.nn.Parameter(weights)
        else:
            w = torch.empty(out_features, in_features, device=device, dtype=dtype)
            std = (2.0/(in_features + out_features))**0.5
            torch.nn.init.trunc_normal_(w, 0,  std, a=-3*std, b=3*std)
            self.weight = torch.nn.Parameter(w)   

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight.T
        return y