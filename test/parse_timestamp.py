import numpy as np
import cv2


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


TIMESTAMP_TEMPL_FILE = './timestamp_templates.xml'

TIME_FORMAT = '{0[0]}{0[1]}:{0[2]}{0[3]}:{0[4]}{0[5]}'
DATE_FORMAT = '{0[0]}{0[1]}-{0[2]}{0[3]}-{0[4]}{0[5]}{0[6]}{0[7]}'


def create_knn(filename):
    knn = cv2.ml.KNearest_create()
    fs_read = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    node_read = fs_read.getNode('opencv_ml_knn')
    knn.read(node_read)
    fs_read.release()
    return knn


TIMESTAMP_MODEL = create_knn(TIMESTAMP_TEMPL_FILE)


def get_timestamp_roi(img, coords, thresh, thresh_type):
    roi = img[coords[0][0]:coords[0][1], coords[1][0]:coords[1][1]]
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


img = cv2.imread('/Users/poxyu/work/metallurg/ex1.png', 0)
time = get_timestamp_roi(img, TIME_COORDS, TIME_THRESH, cv2.THRESH_BINARY_INV)
date = get_timestamp_roi(img, DATE_COORDS, DATE_THRESH, cv2.THRESH_BINARY)

test_time = parse_timestamp(TIMESTAMP_MODEL, time, TIME_DIGITS, TIME_FORMAT)
test_date = parse_timestamp(TIMESTAMP_MODEL, date, DATE_DIGITS, DATE_FORMAT)

print(test_time)
print(test_date)
