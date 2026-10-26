import torch
import math
from linear import Linear
from rope import RotaryPositionalEmbedding

def softmax(logits: torch.Tensor, dim: int):
    """
    e^x_i  / sum_j e^x_j
    """
    # for numerical stability
    torch_max = torch.max(logits, dim=dim, keepdim=True).values
    logits_exp = torch.exp(logits - torch_max)
    sum = logits_exp.sum(dim=dim, keepdim=True)
    return logits_exp / sum


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Given key (K), query (Q), and value (V) tensors, return
        the output of your scaled dot product attention implementation.
    
        Args:
            Q (Float[Tensor, " ... queries d_k"]): Query tensor
            K (Float[Tensor, " ... keys d_k"]): Key tensor
            V (Float[Tensor, " ... keys d_v"]): Values tensor
            mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
        Returns:
            Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    
    scores = Q @ K.transpose(-1, -2)
    scaled = scores / math.sqrt(Q.shape[-1])
    if causal_mask is not None:
        scaled = torch.where(causal_mask, scaled, float('-inf'))

    weights = softmax(scaled, dim=-1)
    return weights @ V

class MultiHeadSelfAttn(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.Wq = Linear(d_model, d_model, device=device, dtype=dtype)
        self.Wk = Linear(d_model, d_model, device=device, dtype=dtype)
        self.Wv = Linear(d_model, d_model, device=device, dtype=dtype)
        self.Wo = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor = None) -> torch.Tensor:
        b, l, d_h = x.shape[0], x.shape[1], self.d_model // self.num_heads
        
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x) # b, l, d_model
        Q = Q.view(b, l, self.num_heads, d_h).transpose(1, 2) # b, h, l, d_h
        K = K.view(b, l, self.num_heads, d_h).transpose(1, 2) # b, h, l, d_h
        V = V.view(b, l, self.num_heads, d_h).transpose(1, 2) # b, h, l, d_h

        if causal_mask is not None:
            causal_mask = causal_mask.unsqueeze(1)  # (b, 1, l, l)
            # mask = mask.expand(-1, self.num_heads, -1, -1)  # (b, num_heads, l, l)
        
        H = scaled_dot_product_attention(Q, K, V, causal_mask) # b X h X l x d_h
        H = H.transpose(1, 2) # b x l x h x d_h
        H = H.flatten(start_dim=2) # b x l x d_model
        
        O = self.Wo(H) # b x l x d_model
        return O
        
class MultiHeadSelfAttnWithRope(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, max_seq_len:int, theta:float, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.Wq = Linear(d_model, d_model, device=device, dtype=dtype)
        self.Wk = Linear(d_model, d_model, device=device, dtype=dtype)
        self.Wv = Linear(d_model, d_model, device=device, dtype=dtype)
        self.Wo = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = RotaryPositionalEmbedding(theta=theta, d_k=d_model//num_heads, max_seq_len=max_seq_len, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor, causal_mask: torch.Tensor = None) -> torch.Tensor:
        b, l, _ = x.shape
        d_h = self.d_model // self.num_heads

        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x) # b, l, d_model

        # split Q, K, V into heads
        Q = Q.view(b, l, self.num_heads, d_h).transpose(1, 2) # (b, h, l, d_h)
        K = K.view(b, l, self.num_heads, d_h).transpose(1, 2) # (b, h, l, d_h)
        V = V.view(b, l, self.num_heads, d_h).transpose(1, 2) # (b, h, l, d_h)

        # aply rope
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        if causal_mask is not None: # (b, l, l)
            causal_mask = causal_mask.unsqueeze(1)  # (b, 1, l, l)

        H = scaled_dot_product_attention(Q, K, V, causal_mask) # (b, h, l, d_h)
        H = H.transpose(1, 2) # (b, l, h, d_h)
        H = H.flatten(start_dim=2) # (b, l, d_model)

        return self.Wo(H) # (b, l, d_model)
        






        


