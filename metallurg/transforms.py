import torch
from torchvision.transforms import Compose

import numpy as np


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


def get_data_transform():
    return Compose([Normalize(), ToTensor()])
