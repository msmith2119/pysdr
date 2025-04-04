import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import hilbert
from scipy.io import wavfile
import simpleaudio as sa
from cmath import phase
import math
from .Utils import *
import scipy as sp


class Signal:
    name = ""
    x = 0
    y = 0
    magz = 0
    phase = 0
    freq = 0
    dt = 0
    size = 0
    fftdone = False

    def __init__(self, name, size, dt):
        self.name = name

        self.size = size
        self.y = np.zeros(size)
        self.x = np.zeros(size)
        self.dt = dt

    def add(self, other):
        s = Signal(self.name + "+" + other.name, self.size, self.dt)
        s.y = self.y + other.y
        s.x = self.x
        return s

    def append(self, offset, y):
        if (offset + y.size > self.size):
            return
        self.y[offset:offset + y.size] = y

    def add_with_offset( self,yin, m):

        end_idx = min(m + yin.size, self.y.size)  # Ensure we don’t go out of bounds
        if m <0:
            self.y[0:end_idx] += yin[-m:end_idx-m]
        else:
            self.y[m:end_idx] += yin[:end_idx - m]  # Add valid portion of yin to y


    def mult(self, other):
        s = Signal(self.name + "*" + other.name, self.size, self.dt)
        s.y = self.y * other.y
        s.x = self.x
        return s

    def square(self):
        s = Signal(self.name + "**2", self.size, self.dt)
        s.y = self.y * self.y
        s.x = self.x
        return s

    def init(self, func):
        for i in range(self.size):
            self.x[i] = self.dt * i
            self.y[i] = func(self.x[i])

    def extract(self, offset, len):
        len = min(self.size - offset, len)
        s = Signal(self.name + "slice", len, self.dt)
        s.x = self.x[offset:offset + len]
        s.y = self.y[offset:offset + len]
        return s

    def fft(self):
        if (self.fftdone):
            return
        fftdone = True
        df = 1 / (self.size * self.dt)
        m = int(self.size / 2)
        self.freq = np.zeros(self.size)
        self.magz = np.zeros(self.size)
        self.phase = np.zeros(self.size)
        self.z = fft(self.y)
        zm = abs(self.z)
        maxval = max(zm)
        for l in range(self.size):
            if (l < m):
                self.magz[l + m] = abs(self.z[l])
                self.phase[l + m] = phase(self.z[l]) * (zm[l] / maxval)
                self.freq[l + m] = df * l

            else:
                self.magz[l - m] = abs(self.z[l])
                self.phase[l - m] = phase(self.z[l]) * (zm[l] / maxval)
                self.freq[l - m] = df * (l - self.size)

    def hilbert(self):
        s = Signal("hilbert(" + self.name + ")", self.size, self.dt)
        s.x = self.x.copy()
        s.y = np.imag(hilbert(self.y))

        return s

    def plotTime(self, stick=False,tmin=0.0,tmax=0.0):

        M = self.size
        P = 0
        plt.figure()
        plt.title(self.name)
        plt.grid(True)
        plt.ylabel("ampl")
        plt.xlabel("sec")
        if tmax != 0 :
            M = int(tmax/self.dt)
        if tmin >0 :
            P = int(tmin/self.dt)
        #xvals = Utils.resize_array(self.x,M)
        #yvals = Utils.resize_array(self.y,M)
        xvals = self.x[P:M]
        yvals = self.y[P:M]
        if stick:
            plt.stem(xvals, yvals)
        else:
            plt.plot(xvals, yvals)

    def plotFreq(self, stick=False, flow=0.0, fhigh=0.0):
        plt.figure()
        self.fft()
        xvals = self.freq
        yvals = (lambda x: 10.0*np.log10(x))(self.magz)
        #yvals = (lambda x: 2*x)(self.magz)

        fn = 1 / (2 * self.dt)
        if fhigh == 0:
            fhigh = fn
        if flow > 0 or fhigh > 0:
            plt.xlim(flow, fhigh)

        plt.title("mag(" + self.name + ")")
        plt.xlabel("freq Hz")
        plt.ylabel("dB")
        plt.grid(True)


        if stick:
            plt.stem(xvals, yvals)
        else:
            plt.plot(xvals, yvals)

    def plotPhase(self, stick=False, logscale=False):
        self.fft()
        plt.figure()
        plt.title("phase(" + self.name + ")")
        if logscale:
            plt.yscale('log')
        if stick:
            plt.stem(self.freq, self.phase, use_line_collection=True)
        else:
            plt.plot(self.freq, self.phase)

    @staticmethod
    def to_file(signals, path):
        fs = int(1.0 / signals[0].dt)
        arrs = []
        for i in range(len(signals)):
            arrs.append(signals[i].y)
        d = np.transpose(np.array(arrs))
        wavfile.write(path, fs, d);

    @staticmethod
    def from_file(path):
        samp_freq, snd = wavfile.read(path)
        y = 0
        if snd.dtype == np.int16:
            y = snd/32767
        else:
            y = snd
        channels = y.shape[1]
        signals = []
        N = len(y[0:, 0])
        dt = 1 / samp_freq
        k = np.array(range(N))
        t = k * dt
        for i in range(channels):
            s = Signal(path + str(i), N, dt)
            signals.append(s)
            s.x = t
            s.y = y[0:, i]
        # np.copyto(s.y,y[0:,i])
        return signals


class PlaySignals:
    @staticmethod
    def play(signals):
        sample_rate = int(1.0 / signals[0].dt)
        nchannels = len(signals)
        N = signals[0].size
        audio_data = np.zeros([nchannels, N])

        for i in range(nchannels):
            audio_data[i] = signals[i].y

        audio_data = audio_data.T

        audio_data *= 32767 / np.max(np.abs(audio_data))
        # convert to 16-bit data
        audio = audio_data.astype(np.int16)
        # start playback
        d = np.zeros([N * nchannels], dtype=np.int16)
        for i in range(N):
            for j in range(nchannels):
                d[nchannels * i + j] = audio[i][j]

        play_obj = sa.play_buffer(d, nchannels, 2, sample_rate)

        # wait for playback to finish before exiting
        play_obj.wait_done()

    @staticmethod
    def playOne(signal):
        signals = []
        signals.append(signal)
        PlaySignals.play(signals)


class SignalSource:
    N = 0
    curpos = 0

    def __init__(self, signal, N):
        self.signal = signal
        self.N = N

    def get(self):
        y = 0
        hi = self.curpos + self.N
        if hi > self.signal.size:
            hi = hi % self.signal.size
            y = np.concatenate(self.signal.y[self.curpos:], self.signal[0:hi])
        else:
            y = self.signal[self.curpos:hi]
        return y
