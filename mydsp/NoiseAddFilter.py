import numpy as np


from mydsp.NoiseSource import NoiseSource, NoiseType
from mydsp.FFTFilter import FFTFilter
from mydsp.Utils import to_number


class NoiseAddFilter(FFTFilter):
    fc = 0
    n = 0
    frame_size =  1
    def __init__(self,name,fs,frame_size,amplitude):
        self.name = name
        self.fs = fs
        self.amplitude = to_number(amplitude)
        self.frame_size = frame_size

        self.prevResult = np.zeros(0)
        self.summary_text = f"Unit Filter @  N={frame_size}"
        self.noise = NoiseSource(NoiseType.WHITE,amplitude,frame_size,1,0)

    def doFrame(self,frame):

        if frame is None:
            return None
        return frame + self.noise.getFrame().ravel()

    def parameters(self):
        return ["amplitude"]

    def set_amplitude(self,amplitude):
        self.amplitude = amplitude
        self.noise = NoiseSource(NoiseType.WHITE, self.amplitude, self.frame_size, 1, 0)
    def amplitude_range(self):
        return [0.0,1.0]
    def summary(self):
        return f"NoiseAddFilter: amplitude={self.amplitude}, fs={self.fs}, frame_size={self.frame_size}"

    @classmethod
    def from_instance(cls, other):
       return cls(other.name , other.fs,other.frame_size,other.amplitude)