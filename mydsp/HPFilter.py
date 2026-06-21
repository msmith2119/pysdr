import numpy as np


from .FFTFilter import FFTFilter
from .Utils import to_number

class HPFilter(FFTFilter):
    description = "HPfilter with parameters: fs=<sampling freq>, fc=<cuttof freq>, sbg=<stopband gain>, frame_size=<frame size>"


    def __init__(self, name, fs, fc,sbg, frame_size):

        self.name = name
        self.fs = to_number(fs)
        self.fc = to_number(fc)
        self.sbg = to_number(sbg)
        self.frame_size = to_number(frame_size)
        self.calc()


    def set_fc(self, fc):
        self.fc = to_number(fc)
        self.calc()
    def set_sbg(self, sbg):
        self.sbg = to_number(sbg)
        self.calc()

    def calc(self):
        self.size = 0
        percentOL = 0.2
        self.overlap = int(percentOL * self.frame_size)

        buffer_size = self.frame_size + self.overlap
        freqs = np.fft.fftfreq(buffer_size, d=1 / self.fs)  # Frequencies for each bin
        self.filt = np.full(buffer_size,self.sbg)  # Start with all-stop

        # Pass everything with |f| <= fc
        self.filt[np.abs(freqs) >= self.fc] = 1.0
    def summary(self):
        return f"HP Filter @ {self.fc} Hz, frame_size={self.frame_size}, fs={self.fs}, fc={self.fc} , sbg={self.sbg}"

    @classmethod
    def from_instance(cls, other):
        return cls(other.name , other.fs,other.fc,other.sbg,other.frame_size)