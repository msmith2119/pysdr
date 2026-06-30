

import numpy as np
from matplotlib import pyplot as plt

from mydsp.FFTFilter import FFTFilter
import ast


from .Utils import to_number, plot_array


class EQFilter(FFTFilter):
    description = "EQ filter with parameters: fs=<sampling freq>, fc=(f1,f2..), gain=(g1,g2..), frame_size=<window size>"


    def __init__(self, name, fs, fc,gain, frame_size):


        self.name = name
        self.fs = to_number(fs)
        self.fc = np.array(fc.strip("()[]").split(), dtype=int)
        self.gain = np.array(gain.strip("()[]").split(), dtype=float)
        self.frame_size = to_number(frame_size)
        percentOL = 0.2
        self.overlap = int(percentOL * self.frame_size)
        self.calc()





    def set_gain(self, gain):
        self.gain = np.array(gain.strip("()[]").split(), dtype=float)
        self.calc()


    def set_band_gain(self,band,gain_value):

        if 0 <= band < len(self.gain):
            self.gain[band] = gain_value
            self.calc()

    def set_dbgain(self,value):
        [idx,val] = value.split(":")
        i = int(idx)
        if 0 <= i < len(self.gain):
            gain = 10 ** (float(val) / 20)
            self.set_band_gain(i,gain)


    def calc(self):


        buffer_size = self.frame_size + self.overlap
        df = self.fs/buffer_size
        fN = self.fs/2.0
        freqs = np.fft.fftfreq(buffer_size, d=1 / self.fs)  # Frequency values for each bin

        self.filt = np.full(buffer_size,1.0)  # Start with all-cut to stop band gain
        n = len(self.fc)

        fl = self.fc[0]-self.fc[0]/2
        fh = self.fc[0] + (self.fc[1]-self.fc[0])/2.0

        mask = ((abs(freqs) >= fl) & (abs(freqs) <= fh))

        self.filt[mask] = self.gain[0]

        fl = self.fc[n-1]  - (self.fc[n-1] - self.fc[n-2])/2.0
        fh = self.fc[n-1] +   (fN - self.fc[n-1])/2.0
        mask = ((abs(freqs) >= fl) & (abs(freqs) <= fh))
        self.filt[mask] = self.gain[n-1]


        for j in range(1,len(self.fc)-1):
            fl = self.fc[j]  - (self.fc[j]-self.fc[j-1])/2.0
            fh = self.fc[j] + (self.fc[j+1]-self.fc[j])/2.0
            mask = ((abs(freqs) >= fl) & (abs(freqs) <= fh))
            self.filt[mask] = self.gain[j]


    def summary(self):
        return f"EQ Filter @ {self.fc} Hz , fc={self.fc}, gain={self.gain} N={self.frame_size}, fs={self.fs}"

    @classmethod
    def from_instance(cls, other):
        return cls(other.name , other.fs, str(other.fc),str(other.gain),other.frame_size)