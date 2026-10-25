import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, weights: None, device=None, dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        if weights is not None:
            self.weights = torch.nn.Parameter(weights)
        else:
            w = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
            torch.nn.init.trunc_normal_(w, 0, 1, a=-3, b=3)
            self.weights = torch.nn.Parameter(w)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.weights[x]
        return y

        

