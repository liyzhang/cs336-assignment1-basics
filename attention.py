import torch
import math

def softmax(logits: torch.Tensor, dim: int):
    """
    e^x_i  / sum_j e^x_j
    """
    # for numerical stability
    torch_max = torch.max(logits, dim=dim, keepdim=True).values
    logits_exp = torch.exp(logits - torch_max)
    sum = logits_exp.sum(dim=dim, keepdim=True)
    return logits_exp / sum


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
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
    if mask is not None:
        scaled = torch.where(mask, scaled, float('-inf'))

    weights = softmax(scaled, dim=-1)
    return weights @ V