from .FFTFilter import FFTFilter
from mydsp.Utils import to_number, shift_freq
from scipy.fftpack import fft, ifft
import numpy as np
from .Parameter import Parameter, ParameterType
from .SincFilter import SincFilter


class FreqShiftFilter:

    description = "FreqShiftFilter fs=<sample_rate>, frame_size=<frame_size>, freq=<freq>"
    def __init__(self,name,fs,frame_size,freq):

        self.name = name
        self.fs = fs
        self.frame_size = frame_size

        self.set_freq(freq)


    def getParameters(self):
        return [Parameter(ParameterType.FLOAT,"freq",-self.fs/20.0,self.fs/20.0,self.freq)]


    def set_freq(self,freq):
        self.freq = to_number(freq)

    def doFrame(self,frame):

        if frame  is None:
            return None

        return shift_freq(frame,self.fs,self.freq)







    @classmethod
    def from_instance(cls,other):
        return cls(other.name, other.fs, other.frame_size, other.freq)

    def summary(self):
        return f"FreqShift frame_size={self.frame_size}, fs={self.fs}, freq={self.freq}"



