from __future__ import annotations

import torch 
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    """

    def __init__(self , d_model :int , eps :float=1e-5) -> None : 
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward (self , x : Tensor)-> Tensor : 

        if x.shape[-1] != self.d_model:
            raise ValueError(f"Input last dimension {x.shape[-1]} does not match d_model {self.d_model}")
        dtype = x.dtype
        x32 = x.float()
        mean_sq = x32.pow(2).mean(dim=-1, keepdim=True)    # (..., 1)
        rms = (mean_sq + self.eps).rsqrt()                # (..., 1)
        y = (x32 * rms).to(dtype) * self.gamma
        return y