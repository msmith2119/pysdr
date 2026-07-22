from mydsp import Parameter
from mydsp.IIRFilter import IIRFilter

from mydsp.Utils import to_number
from .Parameter import Parameter, ParameterType

class RCFilter(IIRFilter):
    def __init__(self,name,fs,frame_size,tau):

        self.name = name
        self.fs = fs
        self.tau = to_number(tau)
        self.frame_size = to_number(frame_size)
        T = 1.0 / self.fs
        a = [T + 2 * self.tau*1e-03, 2 * T, T - 2 * self.tau*1e-03]
        b = [T, 2 * T, T]

        super().__init__(fs, a, b, frame_size)


    def calc(self):
        T = 1.0 / self.fs
        self.a = [T + 2 * self.tau*1e-03, 2 * T, T - 2 * self.tau*1e-03]
        self.b = [T, 2 * T, T]


    def set_tau(self,tau):
        self.tau = to_number(tau)
        self.calc()

    def getParameters(self):
        return  [Parameter(ParameterType.FLOAT,"tau",2000.0/self.fs,300*float(self.frame_size)/self.fs)]

    @classmethod
    def from_instance(cls, other):
        return cls(other.name, other.fs, other.frame_size,other.tau)
    def summary(self):
        return f"RC Filter  tau={self.tau} N={self.frame_size}, fs={self.fs}"