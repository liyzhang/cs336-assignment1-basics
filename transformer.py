import torch
from attention import MultiHeadSelfAttn, MultiHeadSelfAttnWithRope
from swiGlu import SwiGLU
from rmsnorm import RMSNorm
from embedding import Embedding
from linear import Linear

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, eps=1e-5, theta:int | None = None, max_seq_len: int | None = None, device = None):
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

    def forward(self, x: torch.tensor, causal_mask: torch.tensor) -> torch.tensor:
        '''
            x: b, l, d_model
            causal_mask: b, l, l
        '''
        if self.use_rope:
            token_positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0) # token_position: 1 x l, broadcast = b x l
            y = self.attention(
                self.rms_norm_attn(x), token_positions = token_positions, causal_mask=causal_mask
            ) + x # y: b x l x d_model
        else:
            y = self.attention(
                self.rms_norm_attn(x), causal_mask=causal_mask
            ) + x # y: b x l x d_model
        z = self.ffn(self.rms_norm_ffn(y)) + y # z: b x l x d_model
        return z


class TransformerLM(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len: int, num_layers: int, vocab_size: int, eps=1e-5, theta: int = 10000, device=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.theta = theta

        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device)
        self.blocks = torch.nn.ModuleList(
            [TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff, 
                eps=eps,
                theta=theta,
                max_seq_len=max_seq_len,
                device=device
            )  for _ in range(num_layers)]
        )
        self.output_norm = RMSNorm(d_model=d_model, eps=eps, device=device)
        self.output = Linear(in_features=d_model, out_features=vocab_size, device=device)

    def forward(self, x: torch.tensor, causal_mask: torch.tensor) -> torch.tensor:
        '''
            x: b x l 
            causal_mask: b x l x l
        '''
        hidden = self.embedding(x) # hidden: b x l x d_model
        for block in self.blocks:
            hidden = block(hidden, causal_mask) # hidden: b x l x d_model

        hidden = self.output_norm(hidden)
        return self.output(hidden)
        # return softmax(self.output(hidden), dim=-1)
