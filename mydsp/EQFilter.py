

import numpy as np
from matplotlib import pyplot as plt

from mydsp.FFTFilter import FFTFilter
import ast
from .Utils import to_number, plot_array


class EQFilter(FFTFilter):
    description = "EQ filter with parameters: fs=<sampling freq>, fc=(f1,f2..), gain=(g1,g2..), frame_size=<window size>"


    def __init__(self, name, fs, fc,gain, frame_size):

        print(f"passed in fc {fc}")
        self.name = name
        self.fs = to_number(fs)
        self.fc = np.array(fc.strip("()[]").split(), dtype=int)
        self.gain = np.array(gain.strip("()[]").split(), dtype=float)
        self.frame_size = to_number(frame_size)
        percentOL = 0.2
        self.overlap = int(percentOL * self.frame_size)
        self.calc()

        self.summary_text = f"EQ Filter @ {fc} Hz , fc={self.fc}, gain={self.gain} N={frame_size}, fs={fs}"


    def calc(self):


        buffer_size = self.frame_size + self.overlap
        df = self.fs/buffer_size
        freqs = np.fft.fftfreq(buffer_size, d=1 / self.fs)  # Frequency values for each bin
        self.filt = np.full(buffer_size,1.0)  # Start with all-cut to stop band gain
        for i in range(len(self.fc)):
            deltaf = None
            no = int(self.fc[i] / df)
            if i == len(self.fc) - 1:
                deltaf = (self.fc[i] - self.fc[i - 1]) / 2.0
            else:
                deltaf = (self.fc[i + 1] - self.fc[i]) / 2.0

            dn = int(deltaf / df)

            length = 2 * dn + 1
            arr = np.zeros(length)
            for j in range(dn + 1):
                arr[j] = self.gain[i]
            for j in range(dn + 1, length):
                arr[j] = self.gain[i]

            p =   no - dn
            self.filt[p:p + len(arr)] *= arr

        self.filt[buffer_size // 2:] = self.filt[:buffer_size // 2][::-1]

    def summary(self):
        return self.summary_text

    @classmethod
    def from_instance(cls, other):
        return cls(other.name + "cpy", other.fs, str(other.fc),str(other.gain),other.frame_size)