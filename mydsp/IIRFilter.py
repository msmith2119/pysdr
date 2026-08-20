
import numpy as np

import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft
import time

from scipy.signal import lfilter

from mydsp.Utils import plot_array


class IIRFilter:

    def __init__(self,fs,a,b,frame_size):

        self.fs = fs
        self.a = a
        self.b = b
        self.m = max(len(a),len(b))
        self.N = frame_size
        self.profile_data = []
        self.zi = np.zeros(max(len(a), len(b)) - 1)

    def doFrame(self,y):


        if y is None:
            return None

        #start = time.perf_counter()

        yout, self.zi = lfilter(
            self.b,
            self.a,
            y,
            zi=self.zi
        )
        #elapsed = time.perf_counter() - start
        #self.profile_data.append(elapsed)
        return yout



    def plot_impulse(self):

        t = np.arange(self.N)/float(self.fs)
        x = np.zeros(self.N)
        x[0] = 1.0
        y =  self.doFrame(x)

        plt.figure(figsize=(8, 4))
        plt.stem(t, y)
        plt.title("Impulse Response")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plotFFT(self,stick=False):

        N = self.N
        Ny = int(N/2)

        x = np.zeros(self.N)
        x[0] = 1.0
        y = self.doFrame(x)
        # Compute the FFT of the impulse response
        #fft_values = np.fft.fft(impulse_response)
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
        plt.xscale('log')
        plt.grid(True)
        plt.xlim([-self.fs / 2, self.fs / 2])  # Limiting x-axis to show
        #plt.ylim(ymin, ymax)

