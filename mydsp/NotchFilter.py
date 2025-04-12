from mydsp.FFTNotchFilter import FFTNotchFilter


class NotchFilter(FFTNotchFilter):
    description = "Notch filter with parameters: fs=<sampling freq>, fc=<center freq>, fbw=<bandwidth>, N=<window size>"


    def __init__(self, name, fs, fc, fbw, N):
        super().__init__( name, fs, fc, fbw, N)
        self.summary_text = f"Notch Filter @ {fc} Hz (BW {fbw} Hz), N={N}, fs={fs}"

    def summary(self):
        return self.summary_text