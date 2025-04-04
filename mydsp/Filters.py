import numpy as np
import math
import cmath
from  .Utils import *
from .SigClasses import  *
from enum import Enum
from numpy.polynomial.polynomial import polyval
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, ifft

class WindowType(Enum):
    NONE = 0
    HANN = 1
    BRCK = 2

class SampleBuffer:
    depth = 0
    cur_pos = 0
    samples = 0

    def __init__(self, depth):
        self.depth = depth
        self.cur_pos = 0
        self.samples = np.zeros(depth)

    def push(self, v):
        if self.depth < 1:
            return
        self.samples[self.cur_pos] = v
        if self.cur_pos >= self.depth - 1:
            self.cur_pos = 0;
        else:
            self.cur_pos = self.cur_pos + 1

    def push(self, v):

        if self.depth < 1:
            return
        self.samples[self.cur_pos] = v

        self.cur_pos = self.cur_pos + 1
        self.cur_pos = self.cur_pos % self.depth

    def get(self, i):

        if i > self.depth - 1:
            return 0

        if i <= self.cur_pos - 1:
            return self.samples[self.cur_pos - i - 1]
        else:
            return self.samples[self.cur_pos - i - 1 + self.depth]

    def set(self, i, v):

        if i > self.depth - 1:
            return

        if i <= self.cur_pos - 1:
            self.samples[self.cur_pos - i - 1] = v

        else:
            self.samples[self.cur_pos - i - 1 + self.depth] = v


class AnalogFilter:
    name = 0
    a = 0
    b = 0
    fs = 0
    size= 0
    xbuf = 0
    ybuf = 0
    window_type = WindowType.NONE
    overlap = 200
    filt = 0
    prevFrame = np.zeros(1)
    def __init__(self,name, a, b):
        self.name = name
        self.a = a
        self.b = b
        self.xbuf = SampleBuffer(a.size)
        self.ybuf = SampleBuffer(b.size)

    def linear_convolution(self,x,h):

        N = len(x)
        M = len(h)
        y = np.zeros(N)  # Output size matches input size (clamped)

        for n in range(N):
            for m in range(M):
                if n - m >= 0:  # Only accumulate valid indices
                    y[n] += x[n - m] * h[m]

        return y

    def circular_convolution(self,x,hin,hwin):
        N = x.size  # Output size matches x
        y = np.zeros(N, dtype=np.float64)  # Initialize output array
        h = np.pad(hin, (0, N - hin.size), mode='constant')
        for n in range(N):
            for m in range(N):
                y[n] +=  x[m] * h[(n - m) % N]  # Circular indexing using modulo
               # y[n] += hwin[m] * x[m] * h[(n - m) % N]  # Circular indexing using modulo

        return y
    def create_hanning_function(self,M, N):
        assert M > N, "M must be greater than N"

        hanning_window = np.hanning(2 * N)

        def window_function(n):
            if n < N:
                return hanning_window[n]
            elif n < M - N:
                return 1.0
            else:
                return hanning_window[n - M + 2 * N]

        return window_function
    def create_brickwall_function(self,N,M):
        arr = np.zeros(N)  # Initialize with zeros
        arr[M:] = 1.0
        return arr

    def setWindowType(self,wtype):
        self.window_type = wtype

    def getWindow(self,m,n):

        if self.window_type == WindowType.NONE:
            return np.ones(m,dtype=float)
        elif self.window_type == WindowType.HANN:
            hwin = self.create_hanning_function(m, n)
            arr = np.array([hwin(i) for i in range(m)])
            return arr
        else:
            return self.create_brickwall_function(m,n)
        return arr
     #   return np.ones(n)
    def plotWindow(self,m):
        w = self.getWindow(m+self.overlap,self.overlap)

        N = len(w)  # Length of the array
        plt.figure(figsize=(8, 4))  # Set figure size
        plt.plot(np.arange(N), w, marker='o', linestyle='-')  # Plot with markers
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Plot of Array x")
        plt.grid(True)

    def process(self,signal_in):
        signal_out = Signal(self.name + "("+signal_in.name+")",signal_in.size,signal_in.dt)
        for i in range(signal_in.size):
            signal_out.x[i] = signal_in.x[i]
            signal_out.y[i] = self.getOutput(signal_in.y[i])

        return signal_out
    def processFrame(self,frame,hwin):

        M = self.overlap
        yin = frame
        if M != 0:
            yin = np.concatenate((self.prevFrame[-M:], frame))
        #y = self.circular_convolution(yin,self.a,hwin)
        #y = self.circular_convolution(yin,self.a,hwin)
        #y = self.fft_convolution(yin)
        y = self.linear_convolution(yin,self.a)
        self.prevFrame = frame
        #N = frame.size  # Output size matches x
        #y = np.zeros(N, dtype=np.float64)  # Initialize output array
        #h = np.pad(self.a, (0, N - self.a.size), mode='constant')
        #for n in range(N):
         #   for m in range(N):
          #      y[n] += hwin[m]*frame[m] * h[(n - m) % N]  # Circular indexing using modulo

        return y
    def fft_convolution(self,yin):

        zo = fft(yin)
        z1 = zo * self.filt
        yo = ifft(z1).real
        return yo

    def unitFrame(self, frame, hwin):

        N = frame.size  # Output size matches x
        y = np.zeros(N, dtype=np.float64)  # Initialize output array
        h = np.pad(self.a, (0, N - self.a.size), mode='constant')
        for n in range(N):
             y[n] += hwin[n] * frame[n]

        return y
    def processfft(self,signal_in,frame_size):
        hwin = self.getWindow(frame_size,int(self.a.size/2))
        signal_out = Signal(self.name + "(" + signal_in.name + ")", signal_in.size, signal_in.dt)
        num_frames = int(signal_in.size/frame_size)
        if self.prevFrame.size() == 1:
            self.prevFrame = np.zeros(frame_size)
        m = int(frame_size/2)
        filt = fft(self.a,frame_size)
        for i in range(0,num_frames):
            frame = signal_in.y[i*frame_size:(i+1)*frame_size]
            y = self.processFFTFrame(frame,hwin,filt)
            signal_out.append(i*frame_size,y)
        signal_out.x = signal_out.dt*np.array(range(0,signal_out.size))
        return signal_out
    def processConv(self,signal_in,frame_size):
        if self.prevFrame.size == 1:
            self.prevFrame = np.zeros(frame_size)
        M = self.overlap

        hwin = self.getWindow(frame_size + M, M)
        if self.filt == 0 :
            self.filt = fft(self.a, frame_size+M)
        signal_out = Signal(self.name + "(" + signal_in.name + ")", signal_in.size, signal_in.dt)
        num_frames = int(signal_in.size/frame_size)

        for i in range(0,num_frames):

            frame = signal_in.y[i*frame_size:(i+1)*frame_size]
           # yin = np.concatenate((self.prevFrame[-M:], frame))
            yout = self.processFrame(frame,hwin)*hwin
            #yout = self.unitFrame(frame,hwin)
            signal_out.add_with_offset(yout,i*frame_size-M)
            #signal_out.append(i*frame_size,yout)

        signal_out.x = signal_out.dt*np.array(range(0,signal_out.size))
        return signal_out

    def getOutput(self, input):

        sx = 0
        sy = 0

        self.xbuf.push(input)
        self.ybuf.push(0)
        for i in range(self.a.size-1,-1,-1):
            if self.a[i] == 0:
                continue
            sx += self.a[i] * self.xbuf.get(i)

                   # for i in range(1,self.b.size):
        for i in range(self.b.size-1,-1,-1):
            if self.b[i] == 0:
                continue
            sy += self.b[i] * self.ybuf.get(i)
        y = (sx - sy) / self.b[0]
        self.ybuf.set(0, y)
        return y

    def plot(self,df_in,stick=False):

        nsamples = int(self.fs/df_in);
        M = nsamples/2
        fnyquist = self.fs/2;
        magz = np.zeros(nsamples)
        ph = np.zeros(nsamples)
        freq = np.zeros(nsamples)
        for i in range(0,nsamples):
            phi = 2*math.pi*(i-M)/nsamples;
            z = complex(math.cos(phi),-math.sin(phi))
            f = polyval(z,self.a)/polyval(z,self.b)
            magz[i] = abs(f)
            ph[i] = cmath.phase(f)
            freq[i] = -fnyquist + df_in*i

        xvals = freq
#        yvals = (lambda x: 10.0 * np.log10(x))(magz)
        yvals = (lambda x: 10.0 * np.log10(np.clip(x, 1e-10, None)))(magz)

        if stick:
            plt.stem(xvals, yvals, use_line_collection=True)
        else:
            plt.plot(xvals, yvals)
        plt.grid(True)
        plt.xlabel("Hz")
        plt.ylabel("dB")


class SincLPF(AnalogFilter):
    fc = 0
    n = 0

    def __init__(self,name, fs, fc, n):

        m = 2 * n + 1
        self.size = m
        a = np.zeros(m)
        b = np.zeros(1)
        self.fc = fc
        self.fs = fs
        self.n = n
        super().__init__(name,a,b)
        self.window = np.ones(m)
        self.calc()

    def calc(self):

        nu = 2 * self.fc / self.fs
        c = np.zeros(self.n + 1)
        c[0] = nu
        for i in range(1, self.n + 1):
            c[i] = math.sin(nu * i * math.pi) / (i * math.pi)

        for i in range(2 * self.n + 1):
            k = abs(self.n - i)
            self.a[i] = c[k]

        self.b[0] = 1

    def impulse(self):

        s = Signal("impulse-" + self.name, 2 * self.n + 1, 1 / self.fs)
        for i in range(2 * self.n + 1):
            s.x[i] = i / self.fs
            s.y[i] = self.a[i]

        return s

class UnitFilter(AnalogFilter):
    fc = 0
    n = 0

    def __init__(self,name, fs, fc, n):

        m = 2 * n + 1
        self.size = 1
        a = np.zeros(1)
        b = np.zeros(1)
        a[0]=1
        b[0]=1
        self.fc = fc
        self.fs = fs
        self.n = n
        super().__init__(name,a,b)



class RCFilter(AnalogFilter):
    a = 0
    b = 0
    fs = 0
    fc = 0

    def __init__(self,name,fc,fs):
        a = np.zeros(2)
        b = np.zeros(2)
        self.fs = fs
        self.fc = fc
        super().__init__(name, a, b)
        self.calc()

    def calc(self):
        r = self.fc/self.fs
        self.a[0] = 2*math.pi*r
        self.a[1] = self.a[0]
        self.b[0] = 2 + 2*math.pi*r
        self.b[1] = 2 * math.pi * r - 2.0

    def impulse(self,n):
        s = Signal("impulse-" + self.name,  n, 1 / self.fs)
        s.x[0] = 0.0
        s.y[0] = self.getOutput(1.0)
        for i in range(1,n):
            s.x[i] = i/self.fs
            s.y[i] = self.getOutput(0.0)

        return s



class ButterworthFilter(AnalogFilter):
    N = 0
    a = 0
    b = 0
    fs = 0
    fc = 0

    def __init__(self,name,fc,fs,N):
        a = np.zeros(3)
        b = np.zeros(3)
        self.fs = fs
        self.fc = fc
        self.N = N
        super().__init__(name, a, b)
        self.calc()

    def calc(self):
        self.a,self.b = signal.butter(self.N,self.fc,'low',analog=False,fs=self.fs)

    def impulse(self,n):
        s = Signal("impulse-" + self.name,  n, 1 / self.fs)
        s.x[0] = 0.0
        s.y[0] = self.getOutput(1.0)
        for i in range(1,n):
            s.x[i] = i/self.fs
            s.y[i] = self.getOutput(0.0)

        return s
class Echo(AnalogFilter):
    a = 0
    b = 0
    fs = 0
    n = 0
    ff = 0
    fb = 0

    def __init__(self,name,ff,fb,n):
        a = np.zeros(n+1)
        b = np.zeros(n+1)
        self.n = n
        self.ff = ff
        self.fb = fb
        super().__init__(name, a, b)
        self.calc()

    def calc(self):
        self.a[0] = self.ff
        self.a[self.n] = 1.0-self.ff*self.fb
        self.b[0] = 1.0
        self.b[self.n] = -self.fb

class Operations:
    @staticmethod
    def scaledAdd(a,signals):
        cname = ""
        y = np.zeros(signals[0].size)
        for  i in range(len(signals)):
            cname = cname +"+"+ signals[i].name
            y = y + a[i]*signals[i].y
        s = Signal(cname,signals[0].size,signals[0].dt)
        s.y = y
        s.x = signals[0].x.copy()
        return s
