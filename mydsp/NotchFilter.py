import numpy as np

from mydsp.FFTNotchFilter import FFTNotchFilter
from .Utils import to_number

class NotchFilter(FFTNotchFilter):
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

        self.summary_text = f"Notch Filter @ {fc} Hz (BW {fbw} Hz), N={frame_size}, fs={fs}"


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
        return self.summary_text

    @classmethod
    def from_instance(cls, other):
        return cls(other.name + "cpy", other.fs, other.fc, other.fbw,other.frame_size)