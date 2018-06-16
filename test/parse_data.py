import numpy as np
import cv2


# hardcode
DIGIT_WIDTH = 32


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



TEMPLATES = []
TEMPLATES_LABELS = []
for digit in range(10):
    template = cv2.imread('/home/poxyu/work/metallurg/ex1/frames/digits/{0}.png'.format(digit), -1)
    template = template.ravel()
    TEMPLATES.append(template)
    TEMPLATES_LABELS.append([digit])


TEMPLATES = np.array(TEMPLATES).astype(np.float32).astype(np.float32)
TEMPLATES_LABELS = np.array(TEMPLATES_LABELS).astype(np.float32)

knn = cv2.ml.KNearest_create()
knn.setDefaultK(1)
knn.train(TEMPLATES, cv2.ml.ROW_SAMPLE, TEMPLATES_LABELS)




img = cv2.imread('/home/poxyu/work/metallurg/ex1/frames/0.png', 0)

time = img[TIME_COORDS[0][0]:TIME_COORDS[0][1], TIME_COORDS[1][0]:TIME_COORDS[1][1]]
ret, time = cv2.threshold(time, TIME_THRESH, 255, cv2.THRESH_BINARY_INV)
time = time.astype(np.float32)

date = img[DATE_COORDS[0][0]:DATE_COORDS[0][1], DATE_COORDS[1][0]:DATE_COORDS[1][1]]
ret, date = cv2.threshold(date, DATE_THRESH, 255, cv2.THRESH_BINARY)
date = date.astype(np.float32)


date_digits = []
for left, right in DATE_DIGITS:
    digit = date[:, left:right].ravel()
    date_digits.append(digit)

date_digits = np.stack(date_digits)

_, results = knn.predict(date_digits)


time_digits = []
for left, right in TIME_DIGITS:
    digit = time[:, left:right].ravel()
    time_digits.append(digit)

time_digits = np.stack(time_digits)

_, results = knn.predict(time_digits)


knn2 = cv2.ml.KNearest_create()
fs_read = cv2.FileStorage('/home/poxyu/work/metallurg/ex1/frames/digit_templates.xml', cv2.FILE_STORAGE_READ)
arr_read = fs_read.getNode('opencv_ml_knn')
knn2.read(arr_read)
fs_read.release()


last_digit = img[90:134, 726-32:726]
ret, last_digit = cv2.threshold(last_digit, TIME_THRESH, 255, cv2.THRESH_BINARY_INV)
