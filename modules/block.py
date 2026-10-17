from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from modules.rmsnorm import RMSNorm
from modules.mha import MultiHeadAttention
from modules.swiglu import SwiGLU


class TransformerBlock (nn.Module):
    """
    A single LLaMA-style decoder block.

    Structure (Pre-Norm):
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        rope: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, rope=rope)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward (
        self ,
        x : Tensor ,
        mask : Tensor | None = None ,
        positions : Tensor | None = None
    )-> Tensor :
        normed1 = self.norm1(x)
        attn_out = self.attn(normed1, mask=mask, positions=positions)
        x = x + attn_out

        normed2 = self.norm2(x)
        ffn_out = self.ffn(normed2)
        x = x + ffn_out

        return x



