from __future__ import annotations 

import math 

import torch
from torch import Tensor
from modules.softmax import softmax

def sdp_attention(
        q: Tensor , 
        k: Tensor ,
        v: Tensor ,  
        mask : Tensor | None = None      
)-> Tensor:

    if q.dim() < 2 or k.dim() < 2 or v.dim() < 2:
        raise ValueError("q, k, v must have at least 2 dims")
    d_k = q.shape[-1]
    if k.shape[-1] != d_k:
        raise ValueError(f"Dimension mismatch: q has last dimension {d_k}, but k has last dimension {k.shape[-1]}")

    if k.shape[-2] != v.shape[-2]:
        raise ValueError(f"Dimension mismatch: k has second-to-last dimension {k.shape[-2]}, but v has second-to-last dimension {v.shape[-2]}")

    scale = 1.0 / math.sqrt(d_k)

    scores = (q @ k.transpose(-2, -1)) * scale

    if mask is not None : 
        mask = mask.to(dtype=torch.bool)
        neg_inf = torch.full_like(scores, float("-inf"))
        scores = torch.where(mask, scores, neg_inf)

    weights = softmax(scores, dim=-1)
    out = weights @ v

    return out