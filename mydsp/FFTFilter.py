

from .Filters import *

class FFTFilter(AnalogFilter):
    fc = 0
    n = 0

    def __init__(self,name, fs, fc,N):


        self.size = 0
        a =  np.ones(1)
        b =  np.ones(1)
        self.fc = fc

        self.n = 0
        super().__init__(name,fs,a,b)

        freqs = np.fft.fftfreq(N, d=1 / fs)  # Frequencies for each bin
        self.filt = np.zeros(N)  # Start with all-stop

        # Pass everything with |f| <= fc
        self.filt[np.abs(freqs) <= fc] = 1.0

    def impulse(self):

        s = Signal("impulse-" + self.name, self.filt.size, 1 / self.fs)
        y = ifft(self.filt).real
        for i in range(y.size):
            s.x[i] = i / self.fs
            s.y[i] = y[i]

        return s