import sys
import pydub 
import numpy as np
import librosa
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

nFFT = 2048 #  * 4
hLen = 512

def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

import time

class Spectrogram:
    def __init__(self, framerate, y):
        self.framerate, self.y = framerate, y

        if len(self.y.shape) == 2 and self.y.shape[1] == 2:
            self.y = self.y[:, 0]
        
        y = np.float32(self.y) / 2**15
        y = np.abs(librosa.stft(y, hop_length=hLen, n_fft=nFFT))

        self.spec = librosa.amplitude_to_db(y, ref=np.max)
        self.freq = librosa.core.fft_frequencies(sr=framerate, n_fft=nFFT)
        self.time = librosa.core.frames_to_time(
            np.arange(self.spec.shape[1]),
            sr=self.framerate,
            hop_length=hLen,
            n_fft=nFFT
        )

        # 주파수 N에 해당하는 시계열은 N * freq_coef번.
        # N초에 해당하는 프레임은 N * time_coef번.
        self.freq_coef = (len(self.freq) - 1) / int(self.freq[-1])
        self.time_coef = (len(self.time) - 1) / int(self.time[-1])
        
        self.interval = 1000 / self.time_coef

    # 스펙트로그램의 너비(=각 개별 프레임의 길이)
    def spec_width(self):
        return self.framerate // 2

    def frames(self):
        for frame in self.spec.T:
            yield frame

    def freq_size(self): return len(self.freq)
    def time_size(self): return len(self.time)

song_name = sys.argv[1]    

'''
if __name__ == '__main__':
    spec = Spectrogram(*read(song_name))
    fig, ax = plt.subplots()

    M = 32

    ax.set_xlim(0, int(spec.freq[-1]))
    ax.set_ylim(0, 64)

    x = spec.freq
    y = np.zeros((len(x), ))

    line, = plt.plot(x, y)

    def update(frame):
        frame += 80
        line.set_ydata(frame)

        return line,

    ani = FuncAnimation(fig, update, frames=spec.frames(), interval=spec.interval, repeat=False)
    plt.show()
'''

'''
sr, y = read(song_name, normalized=True)
if len(y.shape) == 2 and y.shape[1] == 2:
    y = (y[:, 0] + y[:, 1]) / 2

arr = y[:10 * sr] # 10초

plt.plot(np.linspace(0, 10, 10 * sr), arr)
plt.show()

def update(t):
    frame = arr[
        int(t * sr / 1000):
        int((t + 10) * sr / 1000)
    ]

    line1.set_ydata(frame)

    mag = np.abs(2 * np.fft.fft(frame) / len(frame))
    mag = mag[:len(mag) // 2]

    db = librosa.core.amplitude_to_db(mag)

    line2.set_ydata(db)

    return line1,

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.set_ylim(-1, 1)

ax2.set_ylim(-100, 0)

x1 = np.linspace(0, sr // 100, sr // 100)
y1 = np.zeros((sr // 100, ))

x2 = np.linspace(0, sr // 200, sr // 200)
y2 = np.zeros((sr // 200, ))

line1, = ax1.plot(x1, y1)
line2, = ax2.plot(x2, y2)

ani = FuncAnimation(fig, update, frames=np.arange(0, 10000, 10), interval=10)
plt.show()
'''