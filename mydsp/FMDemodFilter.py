
import numpy as np
from mydsp.Utils import *
import time
class FMDemodFilter:
    description = "FMDemod with parameters: gain=<gain>"
    def __init__(self,name,fs,frame_size,gain):
        self.name = name
        self.fs = fs
        self.frame_size = frame_size
        self.gain = to_number(gain)
        self.last_sample = 0 + 0j
        self.profile_data = []

    def doFrame(self,frame):
        #start = time.perf_counter()
        if frame is None:
           return None

        x = np.concatenate(([self.last_sample], frame))

        y = np.angle(x[1:] * np.conj(x[:-1]))*self.gain

        self.last_sample = frame[-1]
        #elapsed = time.perf_counter() - start
        #self.profile_data.append(elapsed)
        return y

    @classmethod
    def from_instance(cls,other):
        return cls(other.name, other.fs, other.frame_size, other.gain)

    def summary(self):
        return f"FMDemod frame_size={self.frame_size}, fs={self.fs}, gain={self.gain}"