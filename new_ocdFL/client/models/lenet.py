"""
LeNet-5 for MNIST — edge-train-bench style.

This is the model each Jetson trains locally. It's deliberately lightweight
to fit the Orin Nano's 8 GB unified memory comfortably and to keep
communication payloads small (~60 KB state_dict).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNetMNIST(nn.Module):
    """
    Classic LeNet-5 adapted for 28×28 single-channel MNIST images.

    Architecture:
        Conv(1→6, 5) → ReLU → MaxPool(2)
        Conv(6→16, 5) → ReLU → MaxPool(2)
        Flatten → FC(256→120) → ReLU
        FC(120→84) → ReLU
        FC(84→10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
