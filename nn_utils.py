import torch


def softmax(logits: torch.Tensor, dim: int):
    """
    e^x_i  / sum_j e^x_j
    """
    # for numerical stability
    torch_max = torch.max(logits, dim=dim, keepdim=True).values
    logits_exp = torch.exp(logits - torch_max)
    sum = logits_exp.sum(dim=dim, keepdim=True)
    return logits_exp / sum


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.tensor:
    """
    logits: b x vocab_size
    """
    max = torch.max(logits, dim=-1, keepdim=True).values  # b x 1
    sum_exp = torch.exp(logits - max).sum(dim=-1, keepdim=True)  # b x vocab_size
    log_sum_exp = torch.log(sum_exp)  # b x vocab_size

    neg_log_p = torch.gather(log_sum_exp - logits + max, dim=-1, index=targets.unsqueeze(-1))  # b x 1
    return torch.mean(neg_log_p.squeeze(-1))
