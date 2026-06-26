import numpy as np
from matplotlib import pyplot as plt

from mydsp.FFTFilter import FFTFilter

from .Utils import to_number, plot_array


class BPFilter(FFTFilter):
    description = "Band Pass filter with parameters: fs=<sampling freq>, fc=<center freq>, fbw=<bandwidth>, sbg=<stop band gain> frame_size=<window size>"


    def __init__(self, name, fs, fc, fbw,sbg, frame_size):

        self.name = name
        self.fs = to_number(fs)
        self.fc = to_number(fc)
        self.fbw = to_number(fbw)
        self.sbg = to_number(sbg)
        self.frame_size = to_number(frame_size)
        percentOL = 0.2
        self.overlap = int(percentOL * self.frame_size)
        self.calc()



    def set_fc(self, fc):
        self.fc = to_number(fc)
        self.calc()

    def set_sbg(self, sbg):
        self.sbg = to_number(sbg)
        self.calc()
    def set_fbw(self, fbw):
        self.fbw = to_number(fbw)
        self.calc()


    def parameters(self):
        return ["fc","fbw","sbg"]
    def fc_range(self):
        return [0.0,self.fs/2]
    def sbg_range(self):
        return [0.0,1.0]
    def fbw_range(self):
        return [0.1*self.fc,self.fc]
    def calc(self):


        buffer_size = self.frame_size + self.overlap
        freqs = np.fft.fftfreq(buffer_size, d=1 / self.fs)  # Frequency values for each bin
        print(freqs)
        self.filt = np.full(buffer_size,self.sbg)  # Start with all-cut to stop band gain

        # Define the notch bounds
        f_low = self.fc - self.fbw / 2
        f_high = self.fc + self.fbw / 2

        # Zero out bins in the notch range (both positive and negative)
        pass_mask = (np.abs(freqs) >= f_low) & (np.abs(freqs) <= f_high)
        self.filt[pass_mask] = 1.0

    def summary(self):
        return f"BP Filter fc={self.fc} ,fbw={self.fbw} ,sbg={self.sbg} N={self.frame_size}, fs={self.fs}"

    @classmethod
    def from_instance(cls, other):
        return cls(other.name , other.fs, other.fc, other.fbw,other.sbg,other.frame_size)