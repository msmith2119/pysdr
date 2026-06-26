import numpy as np

from mydsp.FFTNotchFilter import FFTNotchFilter
from .FFTFilter import FFTFilter
from .Utils import to_number

class NotchFilter(FFTFilter):
    description = "Notch filter with parameters: fs=<sampling freq>, fc=<center freq>, fbw=<bandwidth>, frame_size=<window size>"


    def __init__(self, name, fs, fc, fbw, frame_size):

        self.name = name
        self.fs = to_number(fs)
        self.fc = to_number(fc)
        self.fbw = to_number(fbw)
        self.frame_size = to_number(frame_size)
        percentOL = 0.2
        self.overlap = int(percentOL * self.frame_size)
        self.calc()



    def parameters(self):
        return ["fc", "fbw"]

    def set_fc(self,fc):
        self.fc = to_number(fc)
        self.calc()

    def set_fbw(self,fbw):
        self.fbw = to_number(fbw)
        self.calc()


    def fc_range(self):
        return [0.0,self.fs/2]


    def fbw_range(self):
        return [0.1*self.fc,self.fc]

    def calc(self):


        buffer_size = self.frame_size + self.overlap
        freqs = np.fft.fftfreq(buffer_size, d=1 / self.fs)  # Frequency values for each bin
        self.filt = np.ones(buffer_size)  # Start with all-pass

        # Define the notch bounds
        f_low = self.fc - self.fbw / 2
        f_high = self.fc + self.fbw / 2

        # Zero out bins in the notch range (both positive and negative)
        notch_mask = (np.abs(freqs) >= f_low) & (np.abs(freqs) <= f_high)
        self.filt[notch_mask] = 0.0
    def summary(self):
        return f"Notch Filter @ {self.fc} Hz (BW {self.fbw} Hz), N={self.frame_size}, fs={self.fs}"

    @classmethod
    def from_instance(cls, other):
        return cls(other.name , other.fs, other.fc, other.fbw,other.frame_size)