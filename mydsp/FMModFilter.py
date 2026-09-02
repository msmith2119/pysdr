from mydsp.Parameter import ParameterType
from mydsp.Utils import to_number
import numpy as np
from .Parameter import Parameter, ParameterType
class FMModFilter:
    description = "FMModFilter fs=<sample_rate> fc=<carrier freq> dev=<frequency deviation>"

    def __init__(self, name,fs,frame_size, fc,dev):

        self.name = name
        self.fs = to_number(fs)
        self.fc = to_number(fc)
        self.dev = to_number(dev)
        self.frame_size = to_number(frame_size)
        self.phase = 0.0

    def doFrame(self,frame):

        if  frame is None:
            return None


        freq = self.fc + self.dev * frame
        dphi = 2.0 * np.pi * freq / self.fs

        phase = self.phase + np.cumsum(dphi)
        self.phase = np.mod(phase[-1], 2 * np.pi)


        return np.exp(1j * phase)


    def set_fc(self,fc):
        self.fc = to_number(fc)

    def set_dev(self,dev):
        self.dev = to_number(dev)


    def getParameters(self):
        return [Parameter(ParameterType.FLOAT,"fc",0.0,0.9*self.fs/2.0,self.fc),
                Parameter(ParameterType.FLOAT,"dev",0,self.fs/2,self.dev)]
    @classmethod
    def from_instance(cls, other):
        return cls(other.name, other.fs, other.frame_size,other.fc, other.dev)

    def summary(self):
        return f"FMModFilter fs={self.fs} fc={self.fc} dev={self.dev}"