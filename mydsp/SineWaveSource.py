
import numpy as np


class SineWaveSource:
    def __init__(self,name,fs,frame_size,num_channels,frequency,amplitude):
        self.name = name
        self.fs = fs
        self.frame_size = frame_size
        self.frequency = frequency
        self.amplitude = amplitude
        self.num_channels = num_channels

        self.df = self.frequency*self.frame_size/self.fs
        self.summary_text = f"SineWave Source fs={self.fs} frame_size={self.frame_size} frequency={self.frequency} amplitude={self.amplitude} num_channels={self.num_channels} "
    description = "source SineWave <name> fs=<fs> frame_size=<frame_size>,num_channels = <num_channels>, frequency=<frequency>, amplitude=<amplitude>"

    def f(self,n):
        return np.sin(self.df * n)

    def getMultiFrame(self):



        a = np.empty((self.frame_size, self.num_channels))
        n = np.arange(self.frame_size)

        values = self.f(n)
        for m in range(self.num_channels):
            a[:, m] = values


        return a

    def close(self):
        return

    def summary(self):
        return self.summary_text
