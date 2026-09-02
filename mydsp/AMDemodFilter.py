

import numpy as np
from mydsp.Utils import *
import time
from .Parameter import ParameterType, Parameter

class AMDemodFilter:
    description = "AMDemod with parameters: gain=<gain>"
    def __init__(self,name,fs,frame_size,gain=1.0):
        self.name = name
        self.fs = fs
        self.frame_size = frame_size
        self.gain = to_number(gain)


    def getParameters(self):
        return [Parameter("gain",0,1,self.gain)]

    def set_gain(self,gain):
        self.gain = to_number(gain)

    def doFrame(self,frame):

        if frame is None:
           return None

        y = np.abs(frame)

        return y

    @classmethod
    def from_instance(cls,other):
        return cls(other.name, other.fs, other.frame_size, other.gain)

    def summary(self):
        return f"AMDemod frame_size={self.frame_size}, fs={self.fs}, gain={self.gain}"