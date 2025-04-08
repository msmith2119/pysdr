
from .Filters import *

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