from .Filters import *

class UnitFilter(AnalogFilter):
    fc = 0
    n = 0

    def __init__(self,name, fs, fc, n):

        m = 2 * n + 1
        self.size = 1
        a = np.zeros(1)
        b = np.zeros(1)
        a[0]=1
        b[0]=1
        self.fc = fc
        self.fs = fs
        self.n = n
        super().__init__(name,a,b)




    def impulse(self,n):
        s = Signal("impulse-" + self.name,  n, 1 / self.fs)
        s.x[0] = 0.0
        s.y[0] = self.getOutput(1.0)
        for i in range(1,n):
            s.x[i] = i/self.fs
            s.y[i] = self.getOutput(0.0)

        return s
