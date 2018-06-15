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

# CORRUPTED_CLASS = 10
# SPACE_CLASS = 11
SPACE_CLASS = 10

# печь
FURNACE_INDEXES = (0, 4)
# тигель
CRUCIBLE_INDEXES = (4, 8)
# проволока
WIRE_INDEXES = (8, 10)
WIRE_FORMAT = '0.{0}'


DEFAULT_FPS = 40
NUM_EVAL_FRAMES = 10

MODEL_FILENAME = './model.pt'
VIDEO_FILENAME = '/Users/poxyu/work/metallurg/ex1.avi'


def get_fps(cap):
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = DEFAULT_FPS
    return fps


def get_digits(frames, data_transform):
    digits = []
    for frame in frames:
        for coords in READOUT_COORDS:
            left_top, right_bottom = coords
            left, top = left_top
            right, bottom = right_bottom
            digit = frame[top:bottom, left:right]
            digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
            digits.append(digit)
    digits = data_transform(np.array(digits))
    return digits


def parse_digits(model, digits):
    inputs = Variable(digits)
    outputs = softmax(model(inputs))
    confidence, predicted = torch.max(outputs.data, 1)
    # reshape
    confidence = confidence.view(NUM_EVAL_FRAMES, -1)
    predicted = predicted.view(NUM_EVAL_FRAMES, -1)
    # max confidence
    confidence, indexes = torch.max(confidence, 0)
    # to numpy
    confidence = confidence.numpy()
    indexes = indexes.numpy()
    predicted = predicted.numpy()
    # best prediction by confidence
    predicted = predicted[indexes, np.arange(predicted.shape[1])]
    return confidence, predicted


def get_readout(predicted, indexes, readout_format='{0}'):
    readout = predicted[indexes[0]:indexes[1]]
    prev_digit = SPACE_CLASS
    result = ''
    for digit in readout:
        if digit == SPACE_CLASS and prev_digit != SPACE_CLASS:
            return 'NULL'
        elif digit != SPACE_CLASS:
            result += str(digit)
        prev_digit = digit
    return readout_format.format(result)


def get_metallurg_net(filename):
    model = MetallurgNet()
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model


def get_data_transform():
    return Compose([Normalize(), ToTensor()])


model = get_metallurg_net(MODEL_FILENAME)
data_transform = get_data_transform()


cap = cv2.VideoCapture(VIDEO_FILENAME)
fps = get_fps(cap)

read_video = True
while read_video:
    frames = []
    for frame_n in range(fps):
        ret, frame = cap.read()
        if ret is False:
            read_video = False
            break
        if frame_n < NUM_EVAL_FRAMES:
            frames.append(frame)
            if frame_n == NUM_EVAL_FRAMES - 1:
                digits = get_digits(frames, data_transform)
                confidence, predicted = parse_digits(model, digits)
                furnace = get_readout(predicted, FURNACE_INDEXES)
                crucible = get_readout(predicted, CRUCIBLE_INDEXES)
                wire = get_readout(predicted, WIRE_INDEXES, WIRE_FORMAT)
                print('{0},{1},{2}'.format(furnace, crucible, wire))


cap.release()



