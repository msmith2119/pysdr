from mydsp.Parameter import ParameterType
import numpy as np

from .Parameter import ParameterType, Parameter
class SquelchFilter:
    def __init__(self,name,fs,frame_size,threshold,mode):
        self.name = name
        self.fs = float(fs)
        self.frame_size = int(frame_size)
        self.threshold = float(threshold)
        self.mode = int(mode)
        self.ms = 0.0
        self.prev = 0.0

    def doFrame(self,frame):

        if frame is None:
            return None

        r2 = self.ms
        self.ms = (self.prev + self.ms + np.mean(frame ** 2)) / 3.0
        self.prev = r2

        rms  = np.sqrt(self.ms)

        if rms < self.threshold:
            if self.mode == 0:
                return np.full(len(frame),0.0)
            else:
                return None
        return frame


    def get_threshold(self):
        return self.threshold

    def set_threshold(self,threshold):
        self.threshold = threshold

    def getParameters(self):
        return [Parameter(ParameterType.FLOAT,"threshold",0.0,1.0,self.threshold)]

    @classmethod
    def from_instance(cls, other):
        return cls(other.name, other.fs, other.frame_size, other.threshold,other.mode)

    def summary(self):
        return f"SquelchFilter frame_size={self.frame_size}, fs={self.fs}, threshold={self.threshold} mode={self.mode}"
