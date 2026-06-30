from .FFTFilter import FFTFilter

from .Filters import *

class UnitFilter(FFTFilter):
    fc = 0
    n = 0
    frame_size =  1
    def __init__(self,name,fs,frame_size):
        self.name = name
        self.fs = fs
        self.frame_size = frame_size
        self.prevResult = np.zeros(0)
        self.summary_text = f"Unit Filter @  N={frame_size}"
    def doFrame(self,frame):

        return frame

    def summary(self):
        return self.summary_text

    @classmethod
    def from_instance(cls, other):
        print(type(other.name))
        return cls(other.name + "cpy", other.fs,other.frame_size)