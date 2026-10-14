from __future__ import annotations
from shutil import which

import torch 

from torch import Tensor

def causal_mask (seq_len : int , device :torch.device| None = None )->Tensor : 
    "mask[i, j] is True if j <= i"

    if seq_len< 2 :
        raise ValueError(" not enough sequence length to create a causal mask")

    ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)

    return torch.tril(ones)