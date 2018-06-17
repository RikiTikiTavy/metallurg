import torch
from torch.autograd import Variable
from torch.nn.functional import softmax

import numpy as np
import cv2

import csv

from metallurg.consts import SPACE_CLASS, READOUT_COORDS


def get_timestamp_roi(img, coords, thresh, thresh_type):
    roi = img[coords[0][0]:coords[0][1], coords[1][0]:coords[1][1]]
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
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
    DEFAULT_FPS = 40
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
            digit = cv2.cvtColor(digit, cv2.COLOR_RGB2GRAY)
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


def write_csv(data, filename):
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)
