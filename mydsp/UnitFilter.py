from .FFTLPFilter import FFTLPFilter
from .Filters import *

class UnitFilter(FFTLPFilter):
    fc = 0
    n = 0
    frame_size =  1
    def __init__(self,name,frame_size):

        self.frame_size = frame_size
        self.prevResult = np.zeros(0)
        self.summary_text = f"Unit Filter @  N={frame_size}"
    def doFrame(self,frame):

        return frame

    def summary(self):
        return self.summary_text