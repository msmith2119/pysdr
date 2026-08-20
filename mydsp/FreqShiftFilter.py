from .FFTFilter import FFTFilter
from mydsp.Utils import to_number
from scipy.fftpack import fft, ifft
import numpy as np
from .Parameter import Parameter, ParameterType
from .SincFilter import SincFilter


class FreqShiftFilter:

    description = "FreqShiftFilter fs=<sample_rate>, frame_size=<frame_size>, freq=<freq>,fc=<cutoff>"
    def __init__(self,name,fs,frame_size,freq,fc,isComplex=False):
        self.name = name
        self.fc = to_number(fc)
        self.fs = fs
        self.freq = to_number(freq)
        self.frame_size = frame_size
        self.bpf = SincFilter("mysinc",self.fs,self.frame_size,201,18000,20000,"BANDPASS")
        self.lpf = SincFilter("mylpf",self.fs,self.frame_size,201,15000,0,"LOWPASS")
        self.isComplex = isComplex



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

    def doFrame(self,frame):

        if frame is None:
            return None


        f19 = self.bpf.doFrame(frame)
        f38 = f19**2
        carrier = 1000*(f38-np.mean(f38))
        #fout = carrier
        fout = self.lpf.doFrame(frame*carrier)

        return fout



    def fft_convolution(self, yin):

        fvals = fft(yin)
        fnew = np.zeros(len(fvals), dtype=complex)
        Ny = int(self.buffer_size/2)
          # Frequencies for each bin
        for i in range(-Ny, Ny - self.p):
            fnew[i] = fvals[i + self.p]*self.z.conjugate()
        for i in range(-Ny + self.p, Ny):
            fnew[i] += fvals[i - self.p]*self.z

        ff = fnew*self.lpfilt
        if self.isComplex:
            return ifft(ff)
        else:
            return ifft(ff).real


    @classmethod
    def from_instance(cls,other):
        return cls(other.name, other.fs, other.frame_size, other.freq,other.fc,other.isComplex)

    def summary(self):
        return f"FreqShift frame_size={self.frame_size}, fs={self.fs}, freq={self.freq} fc={self.fc} isComplex={self.isComplex}"



