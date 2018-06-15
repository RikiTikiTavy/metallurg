import torch
from torch.nn import Conv2d, Module, BatchNorm1d, BatchNorm2d, ReLU, Linear, MaxPool2d, Dropout
from torch.nn.functional import softmax
from torchvision.transforms import Compose
from torch.autograd import Variable

import numpy as np
import cv2


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
    
    def __init__(self, in_, out):
        super(DenseBNRelu, self).__init__()
        self.fc = Linear(in_, out)
        self.bn = BatchNorm1d(out)
        self.activation = ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class MetallurgNet(Module):
    
    def __init__(self):
        super(MetallurgNet, self).__init__()
        
        self.pool = MaxPool2d(2, 2)
        self.dropout = Dropout(inplace=True)
        
        self.conv1 = ConvBNRelu(1, 6)
        self.conv2 = ConvBNRelu(6, 12)
        self.fc1 = DenseBNRelu(12 * 16 * 16, 1024)
        self.fc2 = DenseBNRelu(1024, 128)
        self.fc3 = Linear(128, 12)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, 12 * 16 * 16)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


READOUT_COORDS = [
    ((1465, 858), (1529, 922)), ((1514, 858), (1578, 922)), ((1564, 858), (1628, 922)), ((1613, 858), (1677, 922)),
    ((1465, 906), (1529, 970)), ((1514, 906), (1578, 970)), ((1564, 906), (1628, 970)), ((1613, 906), (1677, 970)),
    ((1564, 1004), (1628, 1068)), ((1613, 1004), (1677, 1068))
]

# READOUT_CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', )



model = MetallurgNet()
model.load_state_dict(torch.load('/home/poxyu/work/metallurg/ex1/algorithm/model.pt'))

model.eval()

data_transform = Compose([
    Normalize(),
    ToTensor()
])


# img = cv2.imread('/home/poxyu/work/metallurg/ex1/algorithm/0.png', 0)
cap = cv2.VideoCapture('/home/poxyu/work/metallurg/ex1/ex1.avi')
_, img = cap.read()
_, img = cap.read()
_, img = cap.read()
_, img = cap.read()
_, img = cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
digits = []
for coords in READOUT_COORDS:
    left_top, right_bottom = coords
    left, top = left_top
    right, bottom = right_bottom
    digit = img[top:bottom, left:right]
    digits.append(digit)

digits = data_transform(np.array(digits))

cap.release()

inputs = Variable(digits)
outputs = softmax(model(inputs))

confidence, predicted = torch.max(outputs.data, 1)
