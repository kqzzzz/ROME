"""Fine-tune the ROME receiver on labeled recordings of received features."""

import argparse
import math
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F

from models import ROME
from recordings import load_recordings


DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "checkpoints"


def optimize(optimizer, logits, targets):
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(logits, targets)
    if not torch.isfinite(loss):
        raise RuntimeError("Training produced a non-finite loss.")
    loss.backward()
    optimizer.step()
    return loss.item()


def train(model, received, labels, levels, args):
    model.to(args.device).eval()
    groups = [(levels == i).nonzero().flatten() for i in range(4)]
    if any(len(group) == 0 for group in groups):
        raise ValueError("Receiver training needs recordings from all four levels.")
    for level in range(1, 4):
        expert = model.experts[level]
        expert.train()
        optimizer = torch.optim.Adam(expert.parameters(), lr=args.lr)
        indices = groups[level]
        for epoch in range(args.epochs):
            total = 0.0
            for batch in indices[torch.randperm(len(indices))].split(args.batch_size):
                logits = expert(received[batch].to(args.device))
                total += len(batch) * optimize(optimizer, logits, labels[batch].to(args.device))
            print(f"Expert {level}, epoch {epoch + 1}: loss {total / len(indices):.4f}")
        expert.eval()

    model.detector.train()
    optimizer = torch.optim.Adam(model.detector.parameters(), lr=args.lr)
    # Each round supplies one batch from each level, including clean signals.
    rounds = math.ceil(len(received) / (4 * args.batch_size))
    for epoch in range(args.epochs):
        total, count = 0.0, 0
        for _ in range(rounds):
            for level in torch.randperm(4).tolist():
                indices = groups[level]
                size = min(args.batch_size, len(indices))
                batch = indices[torch.randint(len(indices), (size,))]
                signal = received[batch].to(args.device)
                with torch.no_grad():
                    probabilities = torch.stack([head(signal).softmax(1) for head in model.experts], dim=1)
                logits = model.detector(signal, probabilities)
                total += size * optimize(optimizer, logits, levels[batch].to(args.device))
                count += size
        print(f"Detector, epoch {epoch + 1}: loss {total / count:.4f}")
    model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", type=Path, help="NPZ with received features, labels, and levels.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=Path("runs/receiver"), help="New checkpoint directory.")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per specialist and for the detector.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(args.weights.resolve()) or output.is_relative_to(DEFAULT_WEIGHTS):
        parser.error("--output must be a new directory outside the pretrained weights.")
    if args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Epochs, batch size, and learning rate must be positive.")
    torch.manual_seed(args.seed)
    received, labels, levels = load_recordings(args.recordings)
    model = ROME.from_pretrained(args.weights)
    train(model, received, labels, levels, args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        pending = Path(staging) / "receiver"
        model.cpu().save_pretrained(pending)
        pending.rename(output)
    print(f"Saved receiver to {output}")


if __name__ == "__main__":
    main()
