from .FFTFilter import FFTFilter
from mydsp.Utils import to_number
from scipy.fftpack import fft, ifft
import numpy as np
from .Parameter import Parameter, ParameterType
from .SincFilter import SincFilter


class FreqShiftFilter(FFTFilter):

    description = "FreqShiftFilter fs=<sample_rate>, frame_size=<frame_size>, freq=<freq>,fc=<cutoff>"
    def __init__(self,name,fs,frame_size,freq,fc,isComplex=False):

        super().__init__(fs,frame_size,isComplex)
        self.name = name
        self.buffer_size = frame_size + self.overlap
        self.freqs = np.fft.fftfreq(self.buffer_size, d=1 / self.fs)
        self.set_freq(freq)
        self.set_fc(fc)





    def getParameters(self):
        return [Parameter("freq",0.1*self.fs/2.0,self.fs/2.0),
                Parameter("fc",0.1*self.fs/2.0,self.fs/2.0)]

    def set_freq(self,freq):
        self.freq = to_number(freq)
        self.p = int(self.buffer_size * self.freq / self.fs)

    def set_fc(self,fc):
        self.fc = to_number(fc)
        self.filt = np.full(self.buffer_size, 0.0)
        self.filt[np.abs(self.freqs) <= self.fc] = 1.0


    def fft_convolution(self, yin):

        fvals = fft(yin)
        fnew = np.zeros(len(fvals), dtype=complex)
        Ny = int(self.buffer_size/2)
          # Frequencies for each bin
        for i in range(-Ny, Ny - self.p):
            fnew[i] = fvals[i + self.p]
        for i in range(-Ny + self.p, Ny):
            fnew[i] += fvals[i - self.p]

        ff = fnew*self.filt
        if self.isComplex:
            return ifft(ff)
        else:
            return ifft(ff).real


    @classmethod
    def from_instance(cls,other):
        return cls(other.name, other.fs, other.frame_size, other.freq,other.fc,other.isComplex)

    def summary(self):
        return f"FreqShift frame_size={self.frame_size}, fs={self.fs}, freq={self.freq} fc={self.fc} isComplex={self.isComplex}"



