from mydsp.FFTLPFilter import FFTLPFilter
from .Utils import to_number

class LPFilter(FFTLPFilter):
    description = "LPfilter with parameters: fs=<sampling freq>, fc=<cuttof freq>, frame_size=<frame size>"


    def __init__(self, name, fs, fc, frame_size):


        fs = to_number(fs)
        fc = to_number(fc)
        frame_size = to_number(frame_size)

        super().__init__( name, fs, fc, frame_size)
        self.summary_text = f"LP Filter @ {fc} Hz, frame_size={frame_size}, fs={fs}"

    def summary(self):
        return self.summary_text