from __future__ import annotations

import argparse
from pathlib import Path

import torch

from modules.data.tokenizer import EOS_TOKEN_ID, decode, encode, get_tokenizer
from modules.kern1 import KERN1


@torch.no_grad()
def sample(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> list[int]:
    """Autoregressively extend (1, T) prompt ids; returns the full id list."""
    for _ in range(max_new_tokens):
        # Only the last max_seq_len tokens can be attended to.
        logits = model(idx[:, -model.max_seq_len:])[:, -1, :]  # (1, V)

        if temperature <= 0:  # greedy
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature

            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))

            if top_p < 1.0:
                # Keep the smallest set whose cumulative probability >= top_p.
                sorted_logits, order = torch.sort(logits, dim=-1, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                remove = (cumprobs - probs) > top_p  # keep the first token that crosses
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.empty_like(sorted_logits).scatter_(-1, order, sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)
        if next_id.item() == EOS_TOKEN_ID:
            break
    return idx[0].tolist()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("runs/kern/checkpoint.pt"))
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    p.add_argument("--top-k", type=int, default=50, help="0 = off")
    p.add_argument("--top-p", type=float, default=0.95, help="1.0 = off")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = KERN1(vocab_size=50257).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.ckpt} (step {ckpt.get('step', '?')}) on{device}")

    tokenizer = get_tokenizer()
    prompt_ids = encode(args.prompt, tokenizer)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    out = sample(model, idx, args.max_new_tokens, args.temperature, args.top_k, args.top_p)

    print("-" * 60)
    print(decode(out, tokenizer))
    print("-" * 60)
    print(f"{args.prompt!r} -> {len(out) - len(prompt_ids)}new tokens")


if __name__ == "__main__":
    main()
