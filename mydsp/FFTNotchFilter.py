

from .Filters import *

class FFTNotchFilter(AnalogFilter):
    fc = 0
    n = 0

    def __init__(self, name, fs, fc, fbw, N):
        a = np.ones(1)
        b = np.ones(1)
        self.fc = fc

        self.n = 0
        super().__init__(name,fs, a, b)

        freqs = np.fft.fftfreq(N, d=1 / fs)  # Frequency values for each bin
        self.filt = np.ones(N)  # Start with all-pass

        # Define the notch bounds
        f_low = fc - fbw / 2
        f_high = fc + fbw / 2

        # Zero out bins in the notch range (both positive and negative)
        notch_mask = (np.abs(freqs) >= f_low) & (np.abs(freqs) <= f_high)
        self.filt[notch_mask] = 0.0

    def impulse(self):

        s = Signal("impulse-" + self.name, self.filt.size, 1 / self.fs)
        y = ifft(self.filt).real
        for i in range(y.size):
            s.x[i] = i / self.fs
            s.y[i] = y[i]

        return s