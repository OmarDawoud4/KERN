from __future__ import annotations

import torch
from torch import Tensor

def softmax(logits : Tensor , dim : int =-1)-> Tensor:
    "numerically stable softmax over  `dim`"

    if not logits.is_floating_point():
        raise TypeError("Input to softmax must be a floating-point tensor")

    if logits.numel() == 0:
        return logits 
    m = logits.max(dim = dim , keepdim = True).values
    shifted = logits -m 

    exp = shifted.exp()
    denom = exp.sum(dim = dim , keepdim = True)

    return exp / denom