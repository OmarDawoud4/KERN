from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    Structure:
        gate = SiLU(W_gate(x))
        up   = W_up(x)
        h    = gate * up
        out  = W_down(h)
    """

    def __init__(self , d_model :int , d_ff:int)->None : 
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w_gate = nn.Linear(d_model, d_ff , bias = False)
        self.w_up = nn.Linear(d_model, d_ff, bias = False)
        self.w_down = nn.Linear(d_ff, d_model, bias = False)

    def forward(self , x : Tensor)-> Tensor :
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Input last dimension {x.shape[-1]} does not match d_model {self.d_model}")
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        h = gate * up
        return  self.w_down(h)
    
