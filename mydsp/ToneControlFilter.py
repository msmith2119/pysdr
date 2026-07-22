from mydsp import Parameter
from mydsp.IIRFilter import IIRFilter

from mydsp.Utils import to_number
from .Parameter import Parameter, ParameterType
import numpy as np

class ToneControlFilter(IIRFilter):
    def __init__(self,name,fs,frame_size,r1,r2):

        self.name = name
        self.fs = to_number(fs)
        self.frame_size = to_number(frame_size)
        T = 1.0 / self.fs
        self.r1 =  to_number(r1)
        self.r2 =  to_number(r2)

        self.a = np.zeros(5)
        self.b = np.zeros(5)
        self.calc()

        super().__init__(fs, self.a, self.b, frame_size)


    def calc(self):
        r1 = self.r1
        r2 = self.r2

        self.a[0]= 4.0e-8 * r1 * r2 + 0.000375 * r1 + 0.00025 * r2 + 0.78125
        self.a[1] = 0.00075 * r1 + 0.0005 * r2 + 3.125
        self.a[2] = -8.0e-8 * r1 * r2 + 4.6875
        self.a[3] = -0.00075 * r1 - 0.0005 * r2 + 3.125
        self.a[4] = 4.0e-8 * r1 * r2 - 0.000375 * r1 - 0.00025 * r2 + 0.78125

        self.b[0] = 4.0e-8 * r1 * r2 + 0.000125 * r1 + 0.00025 * r2 + 0.78125
        self.b[1] = 0.00025 * r1 + 0.0005 * r2 + 3.125
        self.b[2] = -8.0e-8 * r1 * r2 + 4.6875
        self.b[3] = -0.00025 * r1 - 0.0005 * r2 + 3.125
        self.b[4] = 4.0e-8 * r1 * r2 - 0.000125 * r1 - 0.00025 * r2 + 0.78125


    def set_r1(self,r1):
        self.r1 = to_number(r1)
        self.calc()

    def set_r2(self,r2):
        self.r2 = to_number(r2)
        self.calc()

    def getParameters(self):
        return  [Parameter(ParameterType.FLOAT,"r1",1000.0,100000.0),
                 Parameter(ParameterType.FLOAT,"r2",1000.0,100000.0)]

    @classmethod
    def from_instance(cls, other):
        return cls(other.name, other.fs, other.frame_size,other.r1,other.r2)
    def summary(self):
        return f"ToneControl Filter  r1={self.r1} r2={self.r2} N={self.frame_size}, fs={self.fs}"

