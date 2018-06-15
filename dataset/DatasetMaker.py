import sys
import cv2
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, \
    QVBoxLayout, QHBoxLayout, QSizePolicy, \
    QScrollArea, QPushButton, QLineEdit
from PyQt5.QtGui import QPixmap, QImage, QPalette, QMouseEvent
from PyQt5.QtCore import Qt, QPoint, pyqtSignal

FILENAME_VIDEO = '/home/morra/Desktop/Machine_Learning/video.avi'
RESULT_FOLDER = '/home/morra/Desktop/Machine_Learning/dataset/empty'
ROI_SIZE = (64, 64)

if not os.path.exists(RESULT_FOLDER):
    print('Not exist')
    os.makedirs(RESULT_FOLDER)


class ImageLabel(QLabel):
    # custom signal
    # with 2 arguments (x, y)
    # or
    # with 1 argument (QPoint)
    btn_left_press_signal = pyqtSignal([int, int], ['QPoint'])
    btn_left_move_signal = pyqtSignal([int, int], ['QPoint'])
    btn_left_release_signal = pyqtSignal([int, int], ['QPoint'])

    def __init__(self, parent=None):
        super().__init__(parent)
        self.btn_left_pressed = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.btn_left_pressed = True
            # emit signal
            # self.btn_left_press_signal.emit(event.pos()) - QPoint
            self.btn_left_press_signal.emit(event.x(), event.y())

    def mouseMoveEvent(self, event):
        if self.btn_left_pressed:
            # emit signal
            self.btn_left_move_signal.emit(event.x(), event.y())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.btn_left_pressed:
                # emit signal
                self.btn_left_release_signal.emit(event.x(), event.y())
                self.btn_left_pressed = False


class ImageWidget(QWidget):
    frame_changed_signal = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.points = []

        self.init_img_label()
        self.init_scroll_area()
        self.init_layout()
        self.connect_events()

        self.is_making_dataset = False

    def connect_events(self):
        self.img_label.btn_left_press_signal.connect(self.append_point)
        self.img_label.btn_left_move_signal.connect(self.change_point)
        self.img_label.btn_left_release_signal.connect(self.change_point)

    def next_frame(self):
        if self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if ret:
                self.frame = frame
                self.draw_objects()
                self.emit_position()

    def emit_position(self):
        if self.video_cap.isOpened():
            position = self.video_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            self.frame_changed_signal.emit(position)

    def reset_video(self):
        self.change_video_pos(0)

    # value in seconds
    def change_video_pos(self, value):
        if self.video_cap.isOpened():
            value = value * 1000
            ret = self.video_cap.set(cv2.CAP_PROP_POS_MSEC, value)
            if ret:
                self.next_frame()

    def change_point(self, x, y):
        if len(self.points) > 0:
            self.points[len(self.points) - 1] = (x, y)
            self.draw_objects()

    def append_point(self, x, y):
        self.points.append((x, y))
        self.draw_objects()

    def get_rectangles(self):
        rectangles = []
        for point in self.points:
            rect = self.get_rectangle(point)
            rectangles.append(rect)
        return rectangles

    def make_dataset(self):
        if not self.is_making_dataset:
            self.is_making_dataset = True
            print('start make_dataset')
            if self.video_cap.isOpened():
                print('opened')
                position = self.video_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                self.video_cap.set(cv2.CAP_PROP_POS_MSEC, 0)

                rectangles = self.get_rectangles()

                frame_n = 0
                while self.video_cap.isOpened():

                    ret, frame = self.video_cap.read()
                    if ret:
                        print("here")
                        for rect in rectangles:
                            print('lal')
                            left = rect[0][0]
                            top = rect[0][1]
                            right = rect[1][0]
                            bottom = rect[1][1]
                            roi = frame[top:bottom, left:right, :]
                            filename = os.path.join(RESULT_FOLDER, '{0}.png'.format(frame_n))

                            cv2.imwrite(filename, roi)
                            if frame_n % 100 == 0:
                                print('finished with {0} frame'.format(frame_n))
                            frame_n += 1
                    else:
                        break

                self.change_video_pos(position)
            print('end make_dataset')
            self.is_making_dataset = False

    def draw_objects(self):
        frame = self.frame.copy()
        for point in self.points:
            rect = self.get_rectangle(point)
            cv2.rectangle(frame, pt1=rect[0], pt2=rect[1], color=(255, 255, 255), thickness=2)
        self.set_frame(frame)

    def get_rectangle(self, point):
        left = point[0] - ROI_SIZE[0] // 2
        top = point[1] - ROI_SIZE[1] // 2
        right = left + ROI_SIZE[0]
        bottom = top + ROI_SIZE[1]

        rect = ((left, top), (right, bottom))
        return rect

    def init_img_label(self):
        self.img_label = ImageLabel()
        self.img_label.setBackgroundRole(QPalette.Base)
        self.img_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.img_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        # if we need zooming
        # self.img_label.setScaledContents(True)
        # to get coordinates for ROI
        self.img_label.setScaledContents(False)

        self.video_cap = cv2.VideoCapture(FILENAME_VIDEO)
        self.next_frame()

    def reset_frame(self):
        self.points.clear()
        self.set_frame(self.frame)

    def undo_frame(self):
        if len(self.points) > 0:
            self.points.pop()
            self.draw_objects()

    def set_frame(self, new_frame):
        # for gray images
        # qim = QImage(new_frame.data, new_frame.shape[1], new_frame.shape[0], new_frame.strides[0], QImage.Format_Indexed8)
        qim = QImage(new_frame.data, new_frame.shape[1], new_frame.shape[0], new_frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qim)
        self.img_label.setPixmap(pixmap)

    def init_scroll_area(self):
        self.scroll_area = QScrollArea()
        self.scroll_area.setBackgroundRole(QPalette.Dark);
        self.scroll_area.setWidget(self.img_label);
        # self.scroll_area.setVisible(False);
        self.scroll_area.setVisible(True);

    def init_layout(self):
        # create our layout
        layout = QVBoxLayout()
        # # add widgets
        layout.addWidget(self.scroll_area)

        # set the layout of our master widget
        self.setLayout(layout)

    def closeEvent(self, event):
        print('ImageWidget closeEvent')
        self.video_cap.release()


class ControlWidget(QWidget):
    reset_signal = pyqtSignal()
    undo_signal = pyqtSignal()
    reset_video_signal = pyqtSignal()
    change_video_pos_signal = pyqtSignal(float)
    next_frame_signal = pyqtSignal()
    make_dataset_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.btn_reset = QPushButton('Reset')
        self.btn_undo = QPushButton('Undo')
        self.btn_reset_video = QPushButton('Reset Video')
        self.btn_next_frame = QPushButton('Next frame')
        self.btn_change_pos = QPushButton('Change pos')
        # self.cur_position = QLineEdit('0')
        self.cur_position = QLineEdit('')
        self.btn_make_dataset = QPushButton('Make Dataset')

        self.btn_reset.clicked.connect(self.emit_reset)
        self.btn_undo.clicked.connect(self.emit_undo)
        self.btn_reset_video.clicked.connect(self.emit_reset_video)
        self.btn_next_frame.clicked.connect(self.emit_next_frame)
        self.btn_change_pos.clicked.connect(self.emit_change_pos)
        self.btn_make_dataset.clicked.connect(self.emit_make_dataset)

        # create our layout
        # layout = QVBoxLayout()
        layout = QHBoxLayout()
        # # add widgets
        layout.addWidget(self.btn_reset)
        layout.addWidget(self.btn_undo)
        layout.addWidget(self.btn_reset_video)
        layout.addWidget(self.btn_next_frame)
        layout.addWidget(self.btn_change_pos)
        layout.addWidget(self.cur_position)
        layout.addWidget(self.btn_make_dataset)

        # set the layout of our master widget
        self.setLayout(layout)

    def emit_make_dataset(self):
        self.make_dataset_signal.emit()

    def emit_reset(self):
        self.reset_signal.emit()

    def emit_undo(self):
        self.undo_signal.emit()

    def emit_reset_video(self):
        self.reset_video_signal.emit()

    def emit_next_frame(self):
        self.next_frame_signal.emit()

    def emit_change_pos(self):
        value = self.cur_position.text()
        try:
            value = float(value)
            self.change_video_pos_signal.emit(value)
        except:
            print('wrong value')

    def change_cur_position(self, value):
        self.cur_position.setText(str(value))


class MasterWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_widget = ImageWidget()
        self.control_widget = ControlWidget()

        # create our layout
        # layout = QHBoxLayout()
        layout = QVBoxLayout()
        # # add widgets
        # layout.addWidget(self.image_widget, 9)
        # layout.addWidget(self.control_widget, 1)
        layout.addWidget(self.image_widget)
        layout.addWidget(self.control_widget)

        # set the layout of our master widget
        self.setLayout(layout)

        self.connect_events()

        self.image_widget.emit_position()

    def connect_events(self):
        self.control_widget.reset_signal.connect(self.image_widget.reset_frame)
        self.control_widget.undo_signal.connect(self.image_widget.undo_frame)
        self.control_widget.reset_video_signal.connect(self.image_widget.reset_video)
        self.control_widget.change_video_pos_signal.connect(self.image_widget.change_video_pos)
        self.control_widget.next_frame_signal.connect(self.image_widget.next_frame)
        self.image_widget.frame_changed_signal.connect(self.control_widget.change_cur_position)
        self.control_widget.make_dataset_signal.connect(self.image_widget.make_dataset)

    def closeEvent(self, event):
        print('MasterWidget closeEvent')
        self.image_widget.closeEvent(event)


class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)

        # QMainWindow requires a central widget
        self.main_widget = MasterWidget()
        self.setCentralWidget(self.main_widget)

    def closeEvent(self, event):
        print('MainWindow closeEvent')
        self.main_widget.closeEvent(event)


def main():
    # We need to make the QApplication before our QMainWindow
    # We also need to pass in our system argument values (sys.argv)
    app = QApplication(sys.argv)

    main_window = MainWindow()
    main_window.resize(app.primaryScreen().availableSize() * 3 / 5);

    # Show our main window
    main_window.show()
    # Start the event loop processing
    app.exec()


if __name__ == '__main__':
    main()