import torch
from attention import MultiHeadSelfAttn, MultiHeadSelfAttnWithRope
from swiGlu import SwiGLU
from rmsnorm import RMSNorm

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, eps=0.000001, theta:int | None = None, max_seq_len: int | None = None, device = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        
        self.rms_norm_attn = RMSNorm(d_model=self.d_model, eps=eps, device=device)
        if self.theta is None:
            self.attention = MultiHeadSelfAttn(d_model=self.d_model, num_heads=self.num_heads, device=device)
            self.use_rope = False
        else:
            self.attention = MultiHeadSelfAttnWithRope(d_model=self.d_model, num_heads=self.num_heads, max_seq_len=max_seq_len, theta=theta, device=device)
            self.use_rope = True

        self.rms_norm_ffn = RMSNorm(d_model=self.d_model, eps=eps, device=device)
        self.ffn = SwiGLU(d_model=d_model, d_ffn = d_ff, device=device)

    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        if self.use_rope:
            token_positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            y = self.attention(
                self.rms_norm_attn(x), token_positions = token_positions, mask=mask
            ) + x
        else:
            y = self.attention(
                self.rms_norm_attn(x), mask=mask
            ) + x
        z = self.ffn(self.rms_norm_ffn(y)) + y
        return z