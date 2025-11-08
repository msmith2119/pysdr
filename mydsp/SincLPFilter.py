
from .Filters import *

class SincLPFilter(AnalogFilter):
    fc = 0
    n = 0
    description = "Sinc LP Filter with params fs=sampling freq fc=cuttoff freq, n=taps frame_size=size"
    def __init__(self,name,fs, fc, n,frame_size):

        m = 2 * n + 1
        self.size = m
        a = np.zeros(m)
        b = np.zeros(1)
        self.fc = fc
        self.n = n
        super().__init__(name,fs,a,b)

        self.calc()

        self.frame_size = frame_size
        self.overlap = int(self.percentOL*self.frame_size)
        self.summary_text = f"Sinc LPFilter @ {fc} Hz, fs={fs}"

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


    def summary(self):
        return self.summary_text