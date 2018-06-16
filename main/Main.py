import torch
from torch.nn import Conv2d, Module, BatchNorm1d, BatchNorm2d, ReLU, Linear, MaxPool2d, Dropout
from torch.nn.functional import softmax
from torchvision.transforms import Compose
from torch.autograd import Variable

import numpy as np
import cv2
import imageio

import csv


VIDEO_FILENAME = '/home/morra/Desktop/Machine_Learning/video.avi'

DEFAULT_FPS = 40


# left and right borders
TIME_COORDS = ((90, 134), (470, 726))
TIME_THRESH = 200
# left and right borders
TIME_DIGITS = [(0, 32), (32, 64), (96, 128), (128, 160), (192, 224), (224, 256)]

# left and right borders
DATE_COORDS = ((90, 134), (118, 438))
DATE_THRESH = 70
# left and right borders
DATE_DIGITS = [(0, 32), (32, 64), (96, 128), (128, 160), (192, 224), (224, 256), (256, 288), (288, 320)]



TIME_FORMAT = '{0[0]}{0[1]}:{0[2]}{0[3]}:{0[4]}{0[5]}'
DATE_FORMAT = '{0[0]}{0[1]}-{0[2]}{0[3]}-{0[4]}{0[5]}{0[6]}{0[7]}'

READOUT_COORDS = [
    ((1465, 858), (1529, 922)), ((1514, 858), (1578, 922)), ((1564, 858), (1628, 922)), ((1613, 858), (1677, 922)),
    ((1465, 906), (1529, 970)), ((1514, 906), (1578, 970)), ((1564, 906), (1628, 970)), ((1613, 906), (1677, 970)),
    ((1564, 1004), (1628, 1068)), ((1613, 1004), (1677, 1068))
]

SPACE_CLASS = 10

# печь
FURNACE_INDEXES = (0, 4)
# тигель
CRUCIBLE_INDEXES = (4, 8)
# проволока
WIRE_INDEXES = (8, 10)
WIRE_FORMAT = '0.{0}'



NUM_EVAL_FRAMES = 10

TIMESTAMP_FORMAT = '{0} {1}'
RESULT_FORMAT = '{0[0]},{0[1]},{0[2]},{0[3]}'

COLOR_CONVERTER = cv2.COLOR_RGB2GRAY


TIMESTAMP_TEMPL_FILE = './timestamp_templates.xml'
MODEL_FILENAME = './model.pt'

RESULT_FILENAME = './result.csv'

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


def create_knn(filename):
    knn = cv2.ml.KNearest_create()
    fs_read = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    node_read = fs_read.getNode('opencv_ml_knn')
    knn.read(node_read)
    fs_read.release()
    return knn


def get_timestamp_roi(img, coords, thresh, thresh_type):
    roi = img[coords[0][0]:coords[0][1], coords[1][0]:coords[1][1]]
    roi = cv2.cvtColor(roi, COLOR_CONVERTER)
    ret, roi = cv2.threshold(roi, thresh, 255, thresh_type)
    return roi.astype(np.float32)


def parse_timestamp(model, roi, borders, timestamp_format):
    digits = []
    for left, right in borders:
        digit = roi[:, left:right].ravel()
        digits.append(digit)
    digits = np.stack(digits)
    _, results = model.predict(digits)
    results = tuple(results.ravel().astype('int64').tolist())
    return timestamp_format.format(results)


def get_fps(video_reader):
    fps = int(video_reader.get_meta_data()['fps'])
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
            digit = cv2.cvtColor(digit, COLOR_CONVERTER)
            digits.append(digit)
    digits = data_transform(np.array(digits))
    return digits


def parse_digits(model, digits, num_frames):
    inputs = Variable(digits)
    outputs = softmax(model(inputs), dim=1)
    confidence, predicted = torch.max(outputs.data, 1)
    # reshape
    confidence = confidence.view(num_frames, -1)
    predicted = predicted.view(num_frames, -1)
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


def write_csv(data, filename):
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)





def main():
    model = get_metallurg_net(MODEL_FILENAME)
    model_ts = create_knn(TIMESTAMP_TEMPL_FILE)

    data_transform = get_data_transform()

    video_reader = imageio.get_reader(VIDEO_FILENAME)
    fps = get_fps(video_reader)
    total_frames = video_reader.get_length()

    date = None

    csv_data = []

    frames = []
    for idx, frame in enumerate(video_reader):
        mod = idx % fps
        if mod < NUM_EVAL_FRAMES:
            frames.append(frame)
            if (mod == NUM_EVAL_FRAMES - 1) or (idx == total_frames - 1):
                if date is None:
                    date = get_timestamp_roi(frames[-1], DATE_COORDS, DATE_THRESH, cv2.THRESH_BINARY)
                    date = parse_timestamp(model_ts, date, DATE_DIGITS, DATE_FORMAT)

                time = get_timestamp_roi(frames[-1], TIME_COORDS, TIME_THRESH, cv2.THRESH_BINARY_INV)
                time = parse_timestamp(model_ts, time, TIME_DIGITS, TIME_FORMAT)

                timestamp = TIMESTAMP_FORMAT.format(date, time)

                digits = get_digits(frames, data_transform)
                num_frames = len(frames)
                confidence, predicted = parse_digits(model, digits, num_frames)

                furnace = get_readout(predicted, FURNACE_INDEXES)
                crucible = get_readout(predicted, CRUCIBLE_INDEXES)
                wire = get_readout(predicted, WIRE_INDEXES, WIRE_FORMAT)

                csv_line = [timestamp, furnace, crucible, wire]
                csv_data.append(csv_line)
                print(RESULT_FORMAT.format(csv_line))
                frames = []

    write_csv(csv_data, RESULT_FILENAME)


if __name__ == '__main__':
    main()
