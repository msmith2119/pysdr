

from .Filters import *

class Echo(AnalogFilter):
    a = 0
    b = 0
    fs = 0
    n = 0
    ff = 0
    fb = 0

    def __init__(self,name,ff,fb,n):
        a = np.zeros(n+1)
        b = np.zeros(n+1)
        self.n = n
        self.ff = ff
        self.fb = fb
        super().__init__(name, a, b)
        self.calc()

    def calc(self):
        self.a[0] = self.ff
        self.a[self.n] = 1.0-self.ff*self.fb
        self.b[0] = 1.0
        self.b[self.n] = -self.fb
