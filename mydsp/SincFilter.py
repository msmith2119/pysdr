

from enum import Enum
from mydsp.IIRFilter import IIRFilter
from scipy.signal import firwin
import numpy as np
from mydsp.Utils import to_number
from mydsp.Parameter import Parameter,ParameterType


class SincFilter(IIRFilter):
    description = "Sinc filter with parameters: f1=<freq1>, f2=<freq2>, frame_size=<window size> fs = <sampling rate>"

    def __init__(self,name,fs,frame_size,ntaps,f1,f2,ftype):
        self.name = name
        self.fs = to_number(fs)
        self.ntaps = to_number(ntaps)
        self.frame_size = to_number(frame_size)
        self.f1 = to_number(f1)
        self.f2 = to_number(f2)
        self.ftype = ftype
        self.a = None
        self.b = None
        self.a = np.ones(1)
        self.calc()
        super().__init__(fs,self.a,self.b,frame_size)


    def calc(self):

        if self.ftype == "LOWPASS":
            self.b = firwin(self.ntaps, self.f1, fs=self.fs)
        elif self.ftype == "HIGHPASS":
            self.b = firwin(self.ntaps, self.f1, fs=self.fs, pass_zero=False)
        elif self.ftype == "BANDPASS":
            self.b = firwin(self.ntaps, [self.f1, self.f2], fs=self.fs, pass_zero=False)
        elif self.ftype == "NOTCH":
            self.b = firwin(self.ntaps, [self.f1, self.f2], fs=self.fs, pass_zero=True)
        else:
            raise ValueError(f"ftype {self.ftype} not supported")
    def getParameters(self):
        return [Parameter(ParameterType.FLOAT,"f1",0.01*self.fs/2.0,0.9*self.fs/2.0),
                 Parameter(ParameterType.FLOAT, "f2", 0.1 * self.fs / 2.0, 0.9 * self.fs / 2.0)]


    def set_f1(self,f1):
        self.f1 = to_number(f1)
        self.calc()

    def set_f2(self,f2):
        self.f2 = to_number(f2)
        self.calc()

    def summary(self):
        return f"Sinc Filter type = {self.ftype}, f1={self.f1} ,f2={self.f2} , N={self.frame_size}, fs={self.fs}"

    @classmethod
    def from_instance(cls, other):
        print(other.ftype)
        return cls(other.name,other.fs,other.frame_size,other.ntaps,other.f1,other.f2,other.ftype )