
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
        self.power = None
        self.fft = None
        self.n_squared =  float(frame_size)**2
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

        self.fft = fft(frame)
        self.power = (np.abs(self.fft) ** 2)/self.n_squared

        top = heapq.nlargest(5, self.power)
        pavg = np.mean(top)

        if pavg > self.maxpwr:
            self.maxpwr = pavg

        if pavg < self.pwr:
            pavg = 0.9*self.pwr
        self.pwr = (self.pwr + pavg) /2.0

        return frame



    def get_fft(self):
        return self.fft
    def get_power(self):
        return self.power
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