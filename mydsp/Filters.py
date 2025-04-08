import cmath
from enum import Enum

from numpy.polynomial.polynomial import polyval

from .SigClasses import *


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
    filt = np.zeros(1)
    prevFrame = np.zeros(1)
    def __init__(self,name, a, b):
        self.name = name
        self.a = a
        self.b = b
        self.xbuf = SampleBuffer(a.size)
        self.ybuf = SampleBuffer(b.size)



    def setWindowType(self,wtype):
        self.window_type = wtype

    def getEnvelope(self,m,n):


        hwin = create_ola_function(m, n)
        arr = np.array([hwin(i) for i in range(m)])
        return arr


    def plotEnvelope(self,m):
        w = self.getEnvelope(m+self.overlap,self.overlap)

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

    def processFrame(self,frame):

        M = self.overlap
        yin = frame
        if M != 0:
            yin = np.concatenate((self.prevFrame[-M:], frame))

        y = self.fft_convolution(yin)

        self.prevFrame = frame

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

    def processConv(self,signal_in,frame_size):
        if self.prevFrame.size == 1:
            self.prevFrame = np.zeros(frame_size)
        M = self.overlap

        env = self.getEnvelope(frame_size + M, M)
        if self.filt.size == 1 :
            self.filt = fft(self.a, frame_size+M)
        signal_out = Signal(self.name + "(" + signal_in.name + ")", signal_in.size, signal_in.dt)
        num_frames = int(signal_in.size/frame_size)

        for i in range(0,num_frames):

            frame = signal_in.y[i*frame_size:(i+1)*frame_size]
            yout = self.processFrame(frame)*env
            signal_out.add_with_offset(yout,i*frame_size-M)

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

        yvals = (lambda x: 10.0 * np.log10(np.clip(x, 1e-10, None)))(magz)

        if stick:
            plt.stem(xvals, yvals, use_line_collection=True)
        else:
            plt.plot(xvals, yvals)
        plt.grid(True)
        plt.xlabel("Hz")
        plt.ylabel("dB")









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
