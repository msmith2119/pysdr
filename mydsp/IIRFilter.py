
import numpy as np

import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft

from mydsp.Utils import plot_array


class IIRFilter:

    def __init__(self,fs,a,b,frame_size):

        self.fs = fs
        self.a = a
        self.b = b
        self.m = max(a.max_index(),b.max_index())
        self.N = frame_size
        self.xprev = np.zeros(self.m)
        self.yprev = np.zeros(self.m)

    def doFrame(self,y):


        if y is None:
            return None

        xall = np.concatenate((self.xprev, y))
        yall = np.concatenate((self.yprev, np.zeros(self.N)))

        for n in range(self.N):

            idx = self.m + n

            s = 0.0

            # Feedforward
            for k in self.b.keys():
                s += self.b[k] * xall[idx - k]

            # Feedback
            for k in self.a.keys():
                if k == 0:
                    continue
                s -= self.a[k] * yall[idx - k]

            yall[idx] = s / self.a[0]

        self.xprev = y[self.N-self.m:]
        self.yprev = yall[-self.m:]

        return yall[self.m:]



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


    def plotFFT(self,fa,stick=False):

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
        m = int(float(Ny)*fa)
        # Convert FFT values to magnitude and then to dB scale
        magnitude = np.abs(fft_values)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)  # Added small constant to avoid log(0)


        xvals = np.ones(N)

        xvals[:Ny] = freq[Ny:]
        xvals[Ny:] = freq[:Ny]
        yvals = np.ones(N)
        yvals[:Ny] = magnitude_db[Ny:]
        yvals[Ny:] = magnitude_db[:Ny]
        ymax = np.amax(yvals[Ny: Ny+m])
        ymin = np.amin(yvals[Ny:Ny+m])

        # Plot the frequency response
        plt.figure(figsize=(10, 6))
        plt.plot(xvals, yvals)
        plt.title('Frequency Response of the Impulse Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')

        plt.grid(True)
        plt.xlim([-self.fs*fa / 2, self.fs*fa / 2])  # Limiting x-axis to show
        plt.ylim(ymin, ymax)

