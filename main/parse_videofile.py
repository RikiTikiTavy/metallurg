import numpy as np
import torch
from torch.nn import Conv2d, Module, BatchNorm1d, BatchNorm2d, ReLU, Linear


class ToTensor:

    def __call__(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x = torch.from_numpy(x)

        return x


class Normalize:

    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        x = (x / 255).astype(np.float32)

        x -= np.ones(x.shape) * self.mean
        x /= np.ones(x.shape) * self.std

        return x


def conv3x3(in_, out):
    return Conv2d(in_, out, 3, padding=1)


class ConvBNRelu(Module):

    def __init__(self, in_, out):
        super(ConvBNRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.bn = BatchNorm2d(out)
        self.activation = ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class DenseBNRelu(Module):

    def __init__(self, input, output):
        super(DenseBNRelu, self).__init__()
        self.fc = Linear(input, output)
        self.bn = BatchNorm1d(output)
        self.activation = ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


