"""Evaluate ROME on CIFAR-10 over AWGN or on recorded receiver inputs."""

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from models import ROME
from recordings import load_recordings


ROOT = Path(__file__).resolve().parent


@torch.inference_mode()
def evaluate(args):
    model = ROME.from_pretrained(args.weights).to(args.device)
    if args.recordings:
        dataset = TensorDataset(*load_recordings(args.recordings))
    else:
        dataset = CIFAR10(args.data_dir, train=False, download=True, transform=ToTensor())
    count = len(dataset) if args.limit is None else args.limit
    if not 1 <= count <= len(dataset):
        raise ValueError(f"--limit must be between 1 and {len(dataset)}")
    loader = DataLoader(Subset(dataset, range(count)), batch_size=args.batch_size)
    torch.manual_seed(args.seed)
    class_correct = 0
    level_correct = 0
    for inputs, labels, *levels in loader:
        inputs = inputs.to(args.device)
        if args.recordings:
            classes, weights = model.receive(inputs)
            level_correct += (weights.argmax(1).cpu() == levels[0]).sum().item()
        else:
            classes, _ = model(inputs, args.snr)
        class_correct += (classes.argmax(1).cpu() == labels).sum().item()
    print(f"Classification accuracy: {class_correct}/{count} ({100 * class_correct / count:.2f}%)")
    if args.recordings:
        print(f"Level detection accuracy: {level_correct}/{count} ({100 * level_correct / count:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="CIFAR-10 dataset root")
    parser.add_argument("--recordings", type=Path, help="Evaluate an NPZ of labeled received features")
    parser.add_argument("--limit", type=int, help="Evaluate the first N samples (default: all)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--snr", type=float, default=13, help="CIFAR-10 channel SNR in dB")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or not math.isfinite(args.snr):
        parser.error("Batch size must be positive and SNR must be finite.")
    evaluate(args)
