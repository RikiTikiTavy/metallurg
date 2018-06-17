CONFIG_FILENAME = 'config.ini'

VIDEO_FILENAME  = None
RESULT_FILENAME = None

NET_FILENAME = None
KNN_FILENAME = None

TIME_FORMAT         = '{0[0]}{0[1]}:{0[2]}{0[3]}:{0[4]}{0[5]}'
DATE_FORMAT         = '{0[0]}{0[1]}-{0[2]}{0[3]}-{0[4]}{0[5]}{0[6]}{0[7]}'
TIMESTAMP_FORMAT    = '{0} {1}'
WIRE_FORMAT         = '0.{0}'
RESULT_FORMAT       = '{0[0]},{0[1]},{0[2]},{0[3]}'

DATE_THRESH = 70
TIME_THRESH = 200

# left and right borders
TIME_DIGITS = [(0, 32), (32, 64), (96, 128), (128, 160), (192, 224), (224, 256)]
DATE_DIGITS = [(0, 32), (32, 64), (96, 128), (128, 160), (192, 224), (224, 256), (256, 288), (288, 320)]

# left and right borders
DATE_COORDS = ((90, 134), (118, 438))
TIME_COORDS = ((90, 134), (470, 726))
# печь, тигель, проволока
READOUT_COORDS = [
    ((1465, 858), (1529, 922)), ((1514, 858), (1578, 922)), ((1564, 858), (1628, 922)), ((1613, 858), (1677, 922)),  # печь
    ((1465, 906), (1529, 970)), ((1514, 906), (1578, 970)), ((1564, 906), (1628, 970)), ((1613, 906), (1677, 970)),  # тигель
    ((1564, 1004), (1628, 1068)), ((1613, 1004), (1677, 1068))  # проволока
]

FURNACE_INDEXES     = (0, 4)  # печь
CRUCIBLE_INDEXES    = (4, 8)  # тигель
WIRE_INDEXES        = (8, 10)  # проволока

NUM_EVAL_FRAMES = 10

SPACE_CLASS = 10

THRESH_BINARY       = None
THRESH_BINARY_INV   = None


def __read_config():
    import configparser
    import cv2
    import os

    global VIDEO_FILENAME, RESULT_FILENAME
    global THRESH_BINARY, THRESH_BINARY_INV
    global NET_FILENAME, KNN_FILENAME

    config = configparser.ConfigParser()
    config.read(CONFIG_FILENAME)

    VIDEO_FILENAME  = config['DEFAULT']['Video']
    RESULT_FILENAME = config['DEFAULT']['Result']

    THRESH_BINARY       = cv2.THRESH_BINARY
    THRESH_BINARY_INV   = cv2.THRESH_BINARY_INV

    NET_FILENAME = os.path.join('.', *['metallurg', 'models', 'net.pt'])
    KNN_FILENAME = os.path.join('.', *['metallurg', 'models', 'knn.xml'])


__read_config()
del __read_config
