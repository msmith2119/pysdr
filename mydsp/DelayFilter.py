import numpy as np


from .FFTFilter import FFTFilter
from .Parameter import Parameter, ParameterType
from .Utils import to_number
from scipy.fftpack import fft, ifft
from matplotlib import pyplot as plt

class DelayFilter:
    description = "Delay filter with parameters: fs=<sampling freq>,td=<delay>ms , frame_size=<window size>"


    def __init__(self,name,fs,td,frame_size):
        self.name = name
        self.fs = to_number(fs)
        self.td = to_number(td)

        self.frame_size = to_number(frame_size)

        self.calc()



    def getParameters(self):
        return [Parameter(ParameterType.FLOAT,"td",0.0,1000.0*float(self.frame_size)/float(self.fs))]

    def set_td(self,td):
        self.td = to_number(td)
        self.calc()


    def doFrame(self,y):

        if y is None:
            return None

        z = np.zeros(len(y))

        for i in range(len(y)):
            if  i <self.delay_n:
                z[i] = self.delay_buffer[i] + y[i]
            else:
                z[i] = y[i] + y[i-self.delay_n]

        self.delay_buffer = y[-self.delay_n:]

        return z

    def calc(self):

        self.delay_n = min(int((self.td*self.fs)/1000.0),self.frame_size)
        self.delay_buffer = np.zeros(self.delay_n)

    def plotFFT(self,stick=False):


        N = self.frame_size
        Ny = int(N/2)


        # Compute the FFT of the impulse response
        #fft_values = np.fft.fft(impulse_response)

        xin = np.zeros(N)
        xin[0] = 1.0
        y = self.doFrame(xin)
        fft_values = fft(y)
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
        plt.xlim([-self.fs / 2, self.fs / 2])  # Limiting x-axis to show negative and positive frequ
    def summary(self):
        return f"Delay Filter delay = {self.td}, N={self.frame_size}, fs={self.fs}"

    @classmethod
    def from_instance(cls, other):
        return cls(other.name , other.fs, other.td, other.frame_size)
