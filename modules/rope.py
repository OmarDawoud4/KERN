from __future__ import annotations 


import torch 
import torch.nn as nn

from torch import Tensor

class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE), LLaMA half-and-half convention.
    For a token at position p and dimension pair i:
        angle = p * base^(-2i/d_head)
        (x[..., i], x[..., i + d/2])  ->  rotated by angle
    """
     

    def __init__(self , d_head:int , max_seq_len:int = 256 , base: float =10000.0)->None:
          

        super().__init__()

        if d_head % 2 != 0:
               raise ValueError(f"d_head ({d_head}) must be even for RoPE")

        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.base = base


        # inv_freq[i] = base^(-2i/d_head) for i = 0..d_head/2 - 1
        i = torch.arange(0, d_head, 2).float()               # (d/2,)
        inv_freq = 1.0 / (base ** (i / d_head))              # (d/2,)

        self.register_buffer("inv_freq", inv_freq, persistent=False)


        # angles [p,i] = p* inv_freq[i] , shape (T , d/ 2 )

        positions = torch.arange(max_seq_len).float()
        angles = positions [:, None] * inv_freq[None, :]  # (T, d/2)

        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def forward(self , x :Tensor , positions:Tensor|None = None )->Tensor:
        """
        Apply RoPE to the last dim of x.

        Args:
            x: (..., T, d_head)
            positions: optional (T,) int/float tensor. Defaults to 0..T-1.

        Returns:
            Same shape as x, rotated.
        """

        T ,d = x.shape[-2] , x.shape[-1]

        if d != self.d_head:
            raise ValueError(f"Last dimension of x ({d}) must match d_head ({self.d_head})")


        half = d // 2

        if positions is None:
            
            cos = self.cos_cache[:T]   # (T, d/2)
            sin = self.sin_cache[:T]
        else:
            if positions.shape != (T,):
                raise ValueError(
                    f"positions must be shape ({T},), got {tuple(positions.shape)}"
                )
            angles = positions.to(self.cos_cache.dtype)[:, None] * self.inv_freq[None, :]
            cos = angles.cos()
            sin = angles.sin()


        # Cast to x's dtype,then reshape to broadcast against x's leading dims.
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        leading = (1,) * (x.dim() - 2)
        cos = cos.reshape(leading + (T, half))
        sin = sin.reshape(leading + (T, half))

        x1 = x[..., :half]
        x2 = x[..., half:]

        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
             

            

        
