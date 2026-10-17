from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from modules.rmsnorm import RMSNorm
from modules.block import TransformerBlock
from modules.rope import RoPE
from modules.mask import causal_mask


class KERN1(nn.Module):
    """
    KERN-1: A small, LLaMA-style decoder-only transformer.
    """
    def __init__(

        self ,
        vocab_size : int = 65 ,
        d_model : int = 384 , 
        n_heads : int = 6 ,
        d_ff : int = 1024 ,
        n_layers : int = 6 ,
        max_seq_len : int = 256 ,
    )->None :
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size , d_model)
        nn.init.normal_(self.token_embedding.weight, std=0.02)

        #RoPE 
        self.rope = RoPE(d_head = d_model // n_heads , max_seq_len = max_seq_len)

        # 6 Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model = d_model ,
                n_heads = n_heads ,
                d_ff = d_ff ,
                rope = self.rope
            ) for _ in range(n_layers)  
        ])

        # Final Layer Norm
        self.final_norm = RMSNorm(d_model)

        self.head = nn.Linear(d_model , vocab_size , bias = False)


        # weights tying
        self.head.weight = self.token_embedding.weight

        self.register_buffer("causal_mask" , causal_mask(max_seq_len) , persistent = False)


    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids: (B, T) tensor of integers.

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"sequence length {T} exceeds max_seq_len {self.max_seq_len}"
            )

        # 1. Embedding
        x = self.token_embedding(input_ids)  # (B, T, d_model)

        # 2. Blocks (with causal mask slice to match current T)
        mask = self.causal_mask[:T, :T]
        for block in self.blocks:
            x = block(x, mask=mask)

        # 3. Final Norm
        x = self.final_norm(x)

        # 4. Output Head
        logits = self.head(x)  # (B, T, vocab_size)

        return logits

            

