
import numpy as np
import heapq
from scipy.fftpack import fft, ifft
class MeterFilter:


    def __init__(self,name,fs,frame_size):
        self.name = name
        self.fs = float(fs)
        self.frame_size = int(frame_size)
        self.ms = 0.0
        self.pwr = 0.0
        self.maxpwr = 0.0
        self.maxrms = 0.0

    def doFrame(self,frame):
        if frame is None:
            return None

        current_s2 = np.mean(frame ** 2)
        frame_rms = np.sqrt(current_s2)
        if frame_rms > self.maxrms:
            self.maxrms = frame_rms
        if current_s2  < self.ms:
            current_s2 = 0.9*self.ms
        self.ms = (self.ms + current_s2)/2.0

        current_ps = fft(frame)
        power = abs(current_ps**2)
        top = heapq.nlargest(5, power)
        pavg = np.mean(top)

        if pavg > self.maxpwr:
            self.maxpwr = pavg

        if pavg < self.pwr:
            pavg = 0.9*self.pwr
        self.pwr = (self.pwr + pavg) /2.0



        return frame



    def get_rms(self):
        return np.sqrt(self.ms)

    def get_pwr(self):
        return self.pwr

    def set_pwr(self,pwr):
        self.pwr = pwr

    def get_maxpwr(self):
        return self.maxpwr

    def set_maxpwr(self,maxpwr):
        self.maxpwr = maxpwr

    def set_maxrms(self,maxrms):
        self.maxrms = maxrms

    def get_maxrms(self):
        return self.maxrms

    def getParameters(self):
        return []

    def summary(self):
        return f"MeterFilter name = {self.name}, ms = {self.ms}"

    @classmethod
    def from_instance(cls, other):
        return cls(other.name,other.fs,other.frame_size)