from mydsp.Utils import to_number
from scipy.fftpack import fft, ifft
import numpy as np
from .Parameter import Parameter, ParameterType
from .SincFilter import SincFilter

class FMStereoFilter:
    description = "FMStereoFilter fs=<sample_rate>, frame_size=<frame_size>"
    def __init__(self, name, fs, frame_size):
        self.name = name
        self.fs = fs
        self.frame_size = frame_size
        self.bpf = SincFilter("mysinc", self.fs, self.frame_size, 201, 18000, 20000, "BANDPASS")
        self.lpf = SincFilter("mylpf", self.fs, self.frame_size, 201, 15000, 0, "LOWPASS")

    def doFrame(self,frame):

        if frame is None:
            return None

        f19 = self.bpf.doFrame(frame)
        f38 = f19**2
        carrier = 1000*(f38-np.mean(f38))
        fout = self.lpf.doFrame(frame*carrier)
        left_channel =0.5*(frame + fout)
        right_channel = 0.5*(frame - fout)

        yout = left_channel + 1j*right_channel

        return yout

    @classmethod
    def from_instance(cls, other):
        return cls(other.name, other.fs, other.frame_size)

    def summary(self):
        return f"FMStereoFilter frame_size={self.frame_size}, fs={self.fs}"
