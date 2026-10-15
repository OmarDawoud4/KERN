from __future__ import annotations

import torch 
import torch.nn as nn 
from torch import Tensor 



from modules.attention import sdp_attention
from modules.rope import RoPE

class MultiHeadAttention(nn.Module):


    def __init__(self , d_model :int , n_heads : int , rope :RoPE)->None:

        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.rope = rope

        self.w_q = nn.Linear(d_model , d_model , bias = False)
        self.w_k = nn.Linear(d_model , d_model , bias = False)
        self.w_v = nn.Linear(d_model , d_model , bias = False)
        self.w_o = nn.Linear(d_model , d_model , bias = False)


    def forward(
        self ,
        x :Tensor ,
        mask:Tensor | None = None,
        positions :Tensor |None = None )-> Tensor:

        if x.dim() != 3:
            raise ValueError(f"Input x must have 3 dimensions (B, T, D), but got {x.dim()} dimensions")

        B , T , D = x.shape
        if D != self.d_model:
            raise ValueError(f"Input x has last dimension {D}, but d_model is {self.d_model}")

        H , dh = self.n_heads , self.d_k
        #project input to queries, keys, and values to d_model dimensions

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)


        # split into multiple heads 

        # (B , T , D )-> (B , T , H , dh) -> (B , H , T , dh)

        q = q.reshape(B , T , H , dh).transpose(1 , 2)
        k = k.reshape(B , T , H , dh).transpose(1 , 2)
        v = v.reshape(B , T , H , dh).transpose(1 , 2)


        if self.rope is not None:
            # Apply RoPE to q and k
            q = self.rope(q , positions = positions)
            k = self.rope(k , positions = positions)


        out = sdp_attention(q , k , v , mask = mask)

        # Merge heads back to original shape
        # (B , H , T , dh) -> (B , T , H , dh) -> (B , T , D)

        out = out.transpose(1 , 2).reshape(B , T , D)

        return self.w_o(out)
