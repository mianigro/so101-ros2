import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.norm1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.GroupNorm(8, out_channels)

        # Residual connection with down stampling if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(8, out_channels),
            )

        # Initialise layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class CNNEncoder(nn.Module):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        self.input_shape = input_shape
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3)
        self.norm1 = nn.GroupNorm(8, 32)

        # ResNet blocks process each frame
        self.layer1 = self._make_layer(32, 64, stride=2)
        self.layer2 = self._make_layer(64, 128, stride=2)

        # Calculate final spatial dimensions
        h, w = input_shape
        h = math.ceil(h / 8)
        w = math.ceil(w / 8)

        self.flat_dim = 128 * h * w
        self.fc = nn.Linear(self.flat_dim, output_dim)

        # Store the activations for visualization
        self.activations = None

        # Register hook to store layer2 output
        def hook_fn(module, input, output):
            self.activations = output.detach().cpu().numpy()

        self.layer2.register_forward_hook(hook_fn)

        # Initialise weights (same as before)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResBlock(in_channels, out_channels, stride),
            ResBlock(out_channels, out_channels, 1),
        )

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)  # This will trigger the hook and store activations
        x = x.view(x.size(0), -1)
        return self.fc(x)
