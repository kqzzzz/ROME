"""ROME for CIFAR-10, with one shared encoder and four receiver experts."""

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inputs, channels, stride=1, *, detector=False):
        super().__init__()
        def norm():
            if detector:
                return nn.GroupNorm(channels // 2, channels)
            return nn.BatchNorm2d(channels)

        self.conv1 = nn.Conv2d(inputs, channels, 3, stride, 1, bias=False)
        self.bn1 = norm()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = norm()
        self.relu = nn.PReLU() if detector else nn.ReLU()
        self.shortcut = nn.Identity()
        if stride != 1 or inputs != channels:
            self.shortcut = nn.Sequential(nn.Conv2d(inputs, channels, 1, stride, bias=False), norm())

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(y)) + self.shortcut(x))


def stage(inputs, channels, blocks, stride=1, *, detector=False):
    layers = [ResidualBlock(inputs, channels, stride, detector=detector)]
    layers.extend(ResidualBlock(channels, channels, detector=detector) for _ in range(blocks - 1))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(64), nn.ReLU())
        self.layer2 = stage(64, 64, 1)
        self.layer3 = stage(64, 128, 2, 2)
        self.layer4 = stage(128, 256, 2, 2)
        self.layer5 = stage(256, 24, 1)

    def forward(self, images):
        x = self.layer2(self.layer1(images))
        x = self.layer5(self.layer4(self.layer3(x)))
        # Adjacent real values form a complex symbol with unit average power.
        pairs = x.reshape(len(x), -1, 2)
        power = pairs.square().sum(-1).mean(-1).reshape(-1, 1, 1, 1)
        return x / power.clamp_min(1e-12).sqrt()


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer6 = stage(24, 256, 2, 2)
        self.linear = nn.Linear(256, 10)

    def forward(self, received):
        return self.linear(F.avg_pool2d(self.layer6(received), 4).flatten(1))


class Detector(nn.Module):
    """Combine features of the received signal and the four class distributions."""

    def __init__(self):
        super().__init__()
        self.conv1 = stage(24, 256, 2, 2, detector=True)
        self.conv2 = stage(4, 256, 2, detector=True)
        self.linear1 = nn.Linear(512, 4)

    def forward(self, received, probabilities):
        # One 2 x 2 descriptor per expert, in the order used during training.
        stats = torch.stack((probabilities.mean(-1), probabilities.amax(-1),
                             probabilities.amin(-1), probabilities.norm(dim=-1)), dim=-1)
        descriptors = stats.reshape(-1, 4, 2, 2)
        signal = F.avg_pool2d(self.conv1(received), 4).flatten(1)
        confidence = F.avg_pool2d(self.conv2(descriptors), 2).flatten(1)
        return self.linear1(torch.cat((signal, confidence), dim=1))


class ROME(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.experts = nn.ModuleList(Classifier() for _ in range(4))
        self.detector = Detector()

    def receive(self, received):
        """Return class probabilities (N, 10) and level probabilities (N, 4)."""
        scores = torch.stack([expert(received).softmax(1) for expert in self.experts], dim=1)
        levels = self.detector(received, scores).softmax(1)
        return (levels.unsqueeze(-1) * scores).sum(1), levels

    def forward(self, images, snr_db=13.0):
        signal = self.encoder(images)
        sigma = math.sqrt(10 ** (-snr_db / 10) / 2)
        return self.receive(signal + sigma * torch.randn_like(signal))

    def _components(self):
        yield "encoder", self.encoder
        for index, expert in enumerate(self.experts):
            yield f"expert{index}", expert
        yield "detector", self.detector

    @classmethod
    def from_pretrained(cls, path):
        model = cls()
        for name, component in model._components():
            state = torch.load(Path(path) / f"{name}.pt", map_location="cpu", weights_only=True)
            component.load_state_dict(state, strict=True)
        return model.eval()

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name, component in self._components():
            torch.save(component.state_dict(), path / f"{name}.pt")
