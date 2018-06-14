# НА ГОВНОКОД НЕ ОБРАЩАЕМ ВНИМАНИЕ
import sys
import cv2
import os
import shutil
from collections import OrderedDict
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, \
    QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap, QImage, QPalette
from PyQt5.QtCore import Qt, pyqtSignal

# СОЗДАЙТЕ КОПИЮ ДАТАСЕТА, ТАК КАК Я ДЕЛАЮ MOVE() ФАЙЛОВ, А НЕ COPY()
# RAW_FOLDER = '/home/poxyu/work/metallurg/dataset/raw'
RAW_FOLDER = '/home/morra/Desktop/Machine_Learning/dataset/mix'
RESULT_FOLDER = '/home/morra/Desktop/Machine_Learning/dataset/'

if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)


def custom_sort(word):
    numbers = []
    word = int(word.replace('.png', ''))
    return word


# это еще не точно
CLASSES = {
    '0': 0,
    '1': 0,
    '2': 0,
    '3': 0,
    '4': 0,
    '5': 0,
    '6': 0,
    '7': 0,
    '8': 0,
    '9': 0,
    'empty': 0,  # пробелы (зеленый фон)
    'corrupt': 0,  # цифры, но неразличимые (возможно в переходном состоянии)
    'other': 0  # всё остальное
}
CLASSES = OrderedDict(sorted(CLASSES.items()))
# если закрыли прогу и начали заново - проставляем начальный номер файлов
# на основе файлов в папке
for class_name in CLASSES:
    folder = os.path.join(RESULT_FOLDER, class_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        files = os.listdir(folder)
        if len(files) > 0:
            files.sort(key=custom_sort)
            img_num = int(files[len(files) - 1].replace('.png', ''))
            CLASSES[class_name] = img_num + 1


class ImageWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.img_size_divider = 2

        self.file_label = QLabel()
        self.file_label.setAlignment(Qt.AlignCenter)

        self.init_img_label()
        self.init_files()
        self.next_img()

        layout = QVBoxLayout()
        layout.addWidget(self.file_label, 1)
        layout.addWidget(self.img_label, 19)
        self.setLayout(layout)

    def init_img_label(self):
        self.img_label = QLabel()
        print('init label')
        self.img_label.setBackgroundRole(QPalette.Base)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setScaledContents(False)

    def init_files(self):
        self.files = os.listdir(RAW_FOLDER)
        self.files.sort(key=custom_sort, reverse=True)

    def next_img(self):
        if len(self.files) > 0:
            filename = self.files.pop()
            file_label_text = '{0}, {1} images left'.format(filename, len(self.files))
            self.file_label.setText(file_label_text)
            self.filename = os.path.join(RAW_FOLDER, filename)
            self.frame = cv2.imread(self.filename, -1)
            qim = QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], self.frame.strides[0],
                         QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qim)

            w = self.img_label.width() // self.img_size_divider
            h = self.img_label.height() // self.img_size_divider
            self.img_label.setPixmap(pixmap.scaled(w, h, Qt.KeepAspectRatio))
            # self.img_label.setPixmap(pixmap)
        else:
            self.img_label.clear()

    def save_img(self, class_name):
        folder = os.path.join(RESULT_FOLDER, class_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        new_filename = os.path.join(folder, '{0}.png'.format(CLASSES[class_name]))
        shutil.move(self.filename, new_filename)
        CLASSES[class_name] += 1
        self.next_img()

    def resizeEvent(self, event):
        if self.frame is not None:
            qim = QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], self.frame.strides[0],
                         QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qim)

            w = self.img_label.width() // self.img_size_divider
            h = self.img_label.height() // self.img_size_divider
            self.img_label.setPixmap(pixmap.scaled(w, h, Qt.KeepAspectRatio))
        return super().resizeEvent(event)


class ControlWidget(QWidget):
    class_signal = pyqtSignal('QString')

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        row_layout = None
        count = 0
        for class_name in CLASSES:
            btn = QPushButton(class_name)
            btn.clicked.connect(self.make_class_func(class_name))
            if count % 4 == 0:
                row_layout = QHBoxLayout()
                layout.addLayout(row_layout)
            row_layout.addWidget(btn)
            count += 1

        # set the layout of our master widget
        self.setLayout(layout)

    def make_class_func(self, class_name):
        def emit_class():
            self.class_signal.emit(class_name)

        return emit_class


class MasterWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_widget = ImageWidget()
        self.control_widget = ControlWidget()

        # create our layout)
        layout = QVBoxLayout()
        # # add widgets
        layout.addWidget(self.image_widget, 9)
        layout.addWidget(self.control_widget, 1)

        # set the layout of our master widget
        self.setLayout(layout)

        self.connect_events()

    def connect_events(self):
        self.control_widget.class_signal.connect(self.image_widget.save_img)

    def closeEvent(self, event):
        print('MasterWidget closeEvent')
        # self.image_widget.closeEvent(event)


class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)

        # QMainWindow requires a central widget
        self.main_widget = MasterWidget()
        self.setCentralWidget(self.main_widget)

    def closeEvent(self, event):
        print('MainWindow closeEvent')
        # self.main_widget.closeEvent(event)


def main():
    # We need to make the QApplication before our QMainWindow
    # We also need to pass in our system argument values (sys.argv)
    app = QApplication(sys.argv)

    main_window = MainWindow()
    # main_window.resize(app.primaryScreen().availableSize() * 3 / 5);
    main_window.resize(app.primaryScreen().availableSize() / 4);

    # Show our main window
    main_window.show()
    # Start the event loop processing
    app.exec()


if __name__ == '__main__':
    main()