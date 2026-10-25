import torch

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # precompute the rotation matrix for efficiency
        self.rotation_matrix = self._compute_rotation_matrix()

    def _compute_rotation_matrix(self) -> torch.Tensor:
        #  theta_i,k = i/ theta ^ (2*k - 2)/d_k, where k is from 1 to d_k/2
        i = torch.arange(self.max_seq_len, device = self.device) # [L]
        k = torch.arange(self.d_k // 2, device = self.device) # [d/2]
        freqs = self.theta ** ((-2 * k)/ self.d_k) 
        angles = i[:, None] * freqs[None, :]  # [L, d/2]

        # Store the tensor as self.cos_angles.
        # Move it automatically when you call model.to(device).
        # Include it in the model's state_dict() by default.
        # Do not optimize it with gradient descent.
        self.register_buffer("cos_angles", torch.cos(angles))
        self.register_buffer("sin_angles", torch.sin(angles))

    '''
    x: B x Seq_len x d_k
    token_positions: B x Seq_len, token positions is needed because position might not start from 0
    '''
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_angles[token_positions]
        sin = self.sin_angles[token_positions]

        x_even = x[..., 0::2]
        x_odd = x [..., 1::2]

        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos

        y = torch.stack([y_even, y_odd], dim=-1)

        return y.flatten(start_dim=-2)  # B x Seq_len x d_k