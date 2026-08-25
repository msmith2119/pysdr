import numpy as np


from .FFTFilter import FFTFilter
from .Parameter import Parameter, ParameterType
from .Utils import to_number

class LPFilter(FFTFilter):
    description = "LPfilter with parameters: fs=<sampling freq>, fc=<cuttof freq>, sbg=<stopband gain>, frame_size=<frame size>"


    def __init__(self, name, fs, fc,sbg, frame_size,gain=1.0,isComplex="False"):

        self.name = name
        self.fc = to_number(fc)
        self.sbg = to_number(sbg)
        self.gain = to_number(gain)
        super().__init__(fs,frame_size,isComplex)
        self.calc()




    def getParameters(self):
        return [Parameter(ParameterType.FLOAT,"fc",0.0,self.fs/2.0),
                Parameter(ParameterType.FLOAT,"sbg",0.0,1.0)]

    def set_fc(self,fc):
        self.fc = to_number(fc)
        self.calc()
    def set_sbg(self,sbg):
        self.sbg = to_number(sbg)
        self.calc()

    def calc(self):
        self.size = 0


        buffer_size = self.frame_size + self.overlap
        freqs = np.fft.fftfreq(buffer_size, d=1 / self.fs)  # Frequencies for each bin
        self.filt = np.full(buffer_size,self.sbg)  # Start with all-stop


        # Pass everything with |f| <= fc
        self.filt[np.abs(freqs) <= self.fc] = 1.0*self.gain

    def summary(self):
        return f"LP Filter @ {self.fc} Hz, frame_size={self.frame_size}, fs={self.fs}, fc={self.fc} ,gain={self.gain}, sbg={self.sbg}, isComplex={self.isComplex}"

    @classmethod
    def from_instance(cls, other):
        return cls(other.name , other.fs,other.fc,other.sbg,other.frame_size,other.gain,other.isComplex)
