import os
import sys

import numpy as np
import mutagen

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *

from Spectrogram import Spectrogram, read
import Widgets.Visualizer

song_name = sys.argv[1]

width = 16

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self._play = False

        self.initUI()
        
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(song_name)))
        self.player.play()

    def initUI(self):
        def changed(state):
            if state == self.player.StoppedState:
                print(state)

        pixmap = QPixmap()
        metadata = mutagen.File(song_name)
        for tag in metadata.tags.values():
            if tag.FrameID == 'APIC':
                pixmap.loadFromData(tag.data)
                break

        self.visualizer = Widgets.Visualizer.Visualizer(parent=self)
        self.visualizer.setMusic(song_name)

        self.player = QMediaPlayer()
        self.player.stateChanged.connect(changed)
        self.player.positionChanged.connect(self.visualizer.setPosition)
        self.player.setNotifyInterval(50)

        self.setCentralWidget(self.visualizer)

        self.resize(300, 300)
        self.show()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space:
            self._play ^= 1
            if self._play:
                self.player.pause()
                self.visualizer.pause()

            else:
                self.player.play()
                self.visualizer.play()

if __name__ == '__main__':
    os.environ['QT_DEVICE_PIXEL_RATIO'] = '0'
    os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
    os.environ['QT_SCREEN_SCALE_FACTORS'] = '1'
    os.environ['QT_SCALE_FACTOR'] = '1'

    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())