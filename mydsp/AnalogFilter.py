
from mydsp.IIRFilter import IIRFilter
from mydsp.SparseArray import SparseArray

class AnalogFilter(IIRFilter):
    def __init__(self,name,fs,afile,bfile,frame_size):
        self.name = name
        self.fs = fs
        self.frame_size = frame_size
        self.afile = afile
        self.bfile = bfile
        a = SparseArray.load(afile)
        b = SparseArray.load(bfile)
        super().__init__(fs,a,b,frame_size)

    def getParameters(self):
        return  []

    @classmethod
    def from_instance(cls, other):
        return cls(other.name, other.fs, other.afile, other.bfile, other.frame_size)
    def summary(self):
        return f"Analog Filter  afile={self.afile}, bfile = {self.bfile} N={self.frame_size}, fs={self.fs}"
