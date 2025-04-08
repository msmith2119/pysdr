

from .Filters import *

class FFTFilter(AnalogFilter):
    fc = 0
    n = 0

    def __init__(self,name, fs, fc,N):

        m = int(N*fc/fs)
        self.size = m
        a =  np.ones(1)
        b =  np.ones(1)
        self.fc = fc
        self.fs = fs
        self.n = 0
        super().__init__(name,a,b)
        self.filt = create_brick(N,m)



    def impulse(self):

        s = Signal("impulse-" + self.name, self.filt.size, 1 / self.fs)
        y = ifft(self.filt).real
        for i in range(y.size):
            s.x[i] = i / self.fs
            s.y[i] = y[i]

        return s