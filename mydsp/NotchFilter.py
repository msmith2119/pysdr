from mydsp.FFTNotchFilter import FFTNotchFilter
from .Utils import to_number

class NotchFilter(FFTNotchFilter):
    description = "Notch filter with parameters: fs=<sampling freq>, fc=<center freq>, fbw=<bandwidth>, frame_size=<window size>"


    def __init__(self, name, fs, fc, fbw, frame_size):

        fs = to_number(fs)
        fc = to_number(fc)
        fbw = to_number(fbw)
        frame_size = to_number(frame_size)


        super().__init__( name, fs, fc, fbw, frame_size)
        self.summary_text = f"Notch Filter @ {fc} Hz (BW {fbw} Hz), N={frame_size}, fs={fs}"

    def summary(self):
        return self.summary_text