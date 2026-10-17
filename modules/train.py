import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modules.data.dataset import TextDataset
from modules.kern1 import KERN1


def get_device()-> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path: Path, model, opt, step: int, best_val: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "best_val": best_val,
        },
        path,
    )


def load_checkpoint(path: Path, model, opt) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    return ckpt["step"], ckpt["best_val"]


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int | None = None) -> float:
    model.eval()
    total = 0.0
    n = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        ).item()
        n += 1
    model.train()
    return total / max(n, 1)


def main()->None : 
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--out", type=Path, default=Path("runs/kern"))
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-eval-batches", type=int, default=50)
    args = p.parse_args()


    device = get_device()
    train_tokens = torch.load(args.data / "train.pt")
    val_tokens = torch.load(args.data / "val.pt")


    train_ds = TextDataset(train_tokens, args.block_size)
    val_ds = TextDataset(val_tokens, args.block_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) 

    model = KERN1(vocab_size=50257).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_path = args.out / "checkpoint.pt"
    start_step, best_val = 0, float("inf")
    if args.resume and ckpt_path.exists():
        start_step, best_val = load_checkpoint(ckpt_path, model, opt)
        print(f"resumed from step {start_step}, best val {best_val:.4f}")

    print(f"device {device} | params {sum(p.numel() for p in model.parameters()):,}")
    print(f"train examples {len(train_ds):,} | val examples {len(val_ds):,}")



    step = start_step
    running =0.0
    seen = 0
    t0 = time.time()

    while step< args.steps:
        for x , y in train_loader:
            if step >= args.steps:
                break

            x,y = x.to(device), y.to(device)
            logits = model(x)

            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item() 
            seen += 1
            step += 1

            if step % args.eval_every == 0 or step ==args.steps:
                val= evaluate(model , val_loader , device , args.max_eval_batches )
                train_avg = running /max(seen , 1)
                best_val = min(best_val, val)

                print(
                    f"step {step:>6} | train {train_avg:.4f} | val {val:.4f} "
                    f"| {time.time() - t0:6.1f}s"
                )
                running, seen = 0.0, 0

            if step % args.save_every == 0:
                save_checkpoint(ckpt_path, model, opt, step, best_val)

    save_checkpoint(ckpt_path, model, opt, step, best_val)
    print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()


