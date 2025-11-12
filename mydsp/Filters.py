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
    frame_size = 1000
    percentOL = 0.2
    overlap = int(percentOL*frame_size)
    filt = np.zeros(1)
    prevFrame = np.zeros(1)
    prevResult = np.zeros(1)
    envelope = np.zeros(1)

    def __init__(self,name,fs, a, b):
        self.name = name
        self.fs = fs
        self.a = a
        self.b = b
        self.xbuf = SampleBuffer(a.size)
        self.ybuf = SampleBuffer(b.size)


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

    def doFrame(self,frame):

        M = self.overlap
        prev_result_size = self.prevResult.size
        env = self.getEnvelope(self.frame_size + M, M)
        yin = frame
        if M != 0:
            yin = np.concatenate((self.prevFrame[-M:], frame))

        y = self.fft_convolution(yin)*env

        yout = np.copy(self.prevResult)
        yout[-M:0]+=y[:M]
        self.prevResult = y[M:]
        self.prevFrame = frame
# This is the first frame
        if prev_result_size > 1:
            return None
        return yout

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


    def processConv(self,signal_in):


        if self.prevFrame.size == 1:
            self.prevFrame = np.zeros(self.frame_size)
        M = self.overlap

        env = self.getEnvelope(self.frame_size + M, M)
        if self.a.size > 1 :
            self.filt = fft(self.a, self.frame_size+M)
        signal_out = Signal(self.name + "(" + signal_in.name + ")", signal_in.size, signal_in.dt)
        num_frames = int(signal_in.size/self.frame_size)

        for i in range(0,num_frames):

            frame = signal_in.y[i*self.frame_size:(i+1)*self.frame_size]
            yout = self.processFrame(frame)*env
            signal_out.add_with_offset(yout,i*self.frame_size-M)

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



    def plotFFT(self,stick=False):


        N = self.frame_size + self.overlap
        Ny = int(N/2)
        if self.a.size > 1:
            self.filt = fft(self.a , N)

        # Compute the FFT of the impulse response
        #fft_values = np.fft.fft(impulse_response)
        fft_values = self.filt
        # Frequency axis (including negative frequencies)
        freq = np.fft.fftfreq(N, d=1 / self.fs)

        # Convert FFT values to magnitude and then to dB scale
        magnitude = np.abs(fft_values)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)  # Added small constant to avoid log(0)
        xvals = np.ones(N)
        xvals[:Ny] = freq[Ny:]
        xvals[Ny:] = freq[:Ny]
        yvals = np.ones(N)
        yvals[:Ny] = magnitude_db[Ny:]
        yvals[Ny:] = magnitude_db[:Ny]

        # Plot the frequency response
        plt.figure(figsize=(10, 6))
        plt.plot(xvals, yvals)
        plt.title('Frequency Response of the Impulse Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.xlim([-self.fs / 2, self.fs / 2])  # Limiting x-axis to show negative and positive frequencies



    def impulse(self):
        n = self.a.size
        s = Signal("impulse-" + self.name, n , 1 / self.fs)
        for i in range(n ):
            s.x[i] = i / self.fs
            s.y[i] = self.a[i]

        return s






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
