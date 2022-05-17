import numpy as np
import pydub
import librosa
import mutagen

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

__all__ = [
    'Visualizer'
]

nFFT = 2048
hLen = 512

def read(file_name, normalize=False):
    a = pydub.AudioSegment.from_mp3(file_name)
    y = np.array(a.get_array_of_samples())

    if a.channels == 2:
        y = y.reshape((-1, 2))
    
    if normalize:
        return a.frame_rate, np.float32(y) / 2 ** 15
    else:
        return a.frame_rate, y

class Visualizer(QWidget):
    def __init__(self, cols=16, rows=16, parent=None):
        super().__init__(parent)

        self._lock_resize = False

        self._t = 0
        self._amp = 1.0
        self._paused = False

        self.framerate = None
        self.arrays = None

        self.pixmap = None

        self.frame = None

        self.cols = cols
        self.rows = rows

        self.anim = QPropertyAnimation(self, b'amp', self)
        self.anim.setDuration(100)

    @pyqtProperty(float)
    def amp(self):
        return self._amp

    @amp.setter
    def amp(self, _amp):
        self._amp = _amp
        self.update()

    def play(self):
        self._paused = False

        self.anim.setEndValue(1)
        self.anim.start()

        self.update()

    def pause(self):
        self._paused = True

        self.anim.setEndValue(0)
        self.anim.start()

        self.update()

    def setMusic(self, path: str):
        metadata = mutagen.File(path)
        for tag in metadata.tags.values():
            if tag.FrameID == 'APIC':
                self.pixmap = QPixmap()
                self.pixmap.loadFromData(tag.data)

        sr, y = read(path, normalize=True)
        if len(y.shape) == 2 and y.shape[1] == 2:
            y = (y[:, 0] + y[:, 1]) / 2

        self.framerate = sr
        self.arrays = y

    def setPosition(self, position):
        self._t = position
        self.update()

    def _get_frame(self):
        st = self._t * self.framerate // 1000
        ed = (self._t + 10) * self.framerate // 1000

        if ed >= len(self.arrays):
            frame = np.pad(self.arrays[st:len(self.arrays)], (int(10 * self.framerate / 1000), ), constant_values=0)
        else:
            frame = self.arrays[st:ed]

        length = self.framerate // 200

        mag = np.abs(2 * np.fft.fft(frame) / len(frame))[:length]
        
        db = librosa.core.amplitude_to_db(mag) + 80
        db *= (self.rows / 80)

        mat = np.zeros((length, self.cols))
        asp = length // self.cols

        for i, j in np.ndindex(self.cols, asp):
            mat[i * asp + j, i] = 1 / asp

        res = np.matmul(db, mat).astype(int)

        return res
    '''
    def resizeEvent(self, event: QResizeEvent):
        if self._lock_resize: super().resizeEvent(event)

        self._lock_resize = True

        width = event.size().width()
        height = event.size().height()

        if width > height: self.resize(height, height)
        else: self.resize(width, width)

        self._lock_resize = False
    '''
    def paintEvent(self, event: QPaintEvent):
        if self.arrays is None: return
        if self.height() == 0 or self.width() == 0: return
        
        r_w = self.width() / (self.cols - 1)
        r_h = self.height() / (self.rows - 1)

        asp = self.width() / self.height()

        p_w = min(self.pixmap.width(), self.pixmap.height() * asp)
        p_h = min(self.pixmap.height(), self.pixmap.width() / asp)
        p_x = (self.pixmap.width() - p_w) // 2
        p_y = (self.pixmap.height() - p_h) // 2

        with QPainter(self) as painter:
            painter.setRenderHint(QPainter.Antialiasing)
            if self.pixmap is not None:
                painter.drawPixmap(self.rect(), self.pixmap, QRect(p_x, p_y, p_w, p_h))

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor.fromRgba(0x7F000000))

            for i, v in enumerate(self._get_frame()):
                v = int(v * self._amp)

                for j in range(v):
                    p = self.rows - j - 1
                    
                    rect = QRectF(r_w * i, r_h * p, r_w, r_h)
                    rect.adjust(1, 1, -1, -1)

                    painter.drawRect(rect)