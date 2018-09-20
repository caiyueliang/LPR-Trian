# coding=utf-8
import cv2
import sys
import os
import time
import common as common

import sys
from PyQt5.QtWidgets import QWidget, QPushButton, QDesktopWidget, QApplication
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QHBoxLayout, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QCoreApplication


class Example(QWidget):

    def __init__(self):
        super(Example, self).__init__()
        self.w = 800
        self.h = 600
        self.init_ui()

    def init_ui(self):
        self.resize(self.w, self.h)
        self.center()
        # self.move(300, 300)

        self.show_image('/home/caiyueliang/deeplearning/Data/car_recognition/train/blue_2/480467_闽D77V56_0.png')
        self.set_quit_button()      # 退出按钮

        self.setWindowTitle('Sign Ocr')
        self.show()

    # =========================================================================
    # 控制窗口显示在屏幕中心的方法
    def center(self):
        # 获得窗口
        qr = self.frameGeometry()
        # 获得屏幕中心点
        cp = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def show_image(self, image_path):
        hbox = QHBoxLayout(self)
        pixmap = QPixmap(image_path)

        lbl = QLabel(self)
        lbl.setPixmap(pixmap)
        lbl.resize(300, 200)

        hbox.addWidget(lbl)
        self.setLayout(hbox)

        self.move(300, 200)
        self.setWindowTitle('Red Rock')
        self.show()

    # =========================================================================
    def set_quit_button(self):
        qbtn = QPushButton('Quit', self)
        qbtn.clicked.connect(QCoreApplication.instance().quit)
        qbtn.resize(qbtn.sizeHint())
        qbtn.move(self.w - 100, self.h - 100)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', "Are you sure to quit?", QMessageBox.Yes
                                     | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
