"""
LeNet-5 for MNIST-family and CIFAR-10 — edge-train-bench style.

This is the model each Jetson trains locally. It's deliberately lightweight
to fit the Orin Nano's 8 GB unified memory comfortably and to keep
communication payloads small (~60 KB state_dict).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNetMNIST(nn.Module):
    """
    LeNet-5 supporting both MNIST-family (28×28 grayscale) and CIFAR-10 (32×32 RGB).

    Architecture:
        Conv(in_channels→6, 5) → ReLU → MaxPool(2)
        Conv(6→16, 5)          → ReLU → AdaptiveAvgPool(5×5)
        Flatten → FC(400→120) → ReLU
        FC(120→84) → ReLU
        FC(84→num_classes)

    AdaptiveAvgPool2d after conv2 fixes the spatial size at 5×5 regardless of
    whether the input is 28×28 (MNIST/FashionMNIST) or 32×32 (CIFAR-10), so
    the FC layers are identical for both datasets.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, (5, 5))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
