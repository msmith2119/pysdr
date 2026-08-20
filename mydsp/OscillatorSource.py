from mydsp.Utils import to_number
import numpy as np

class OscillatorSource:
    Params = ["sample_rate","frame_size","frequency","amplitude"]
    description = "Oscillator Source, sample_rate=<sample_rate> frame_size=<frame_size>, frequency=<frequency>, amplitude=<amplitude>, phase=<phase>, output=<output>"
    def __init__(self,name,sample_rate,frame_size,frequency,amplitude,phase=0.0,num_frames = 0,output="sin"):
        self.name = name
        self.fs = to_number(sample_rate)
        self.frame_size = to_number(frame_size)
        self.frequency = to_number(frequency)
        self.amplitude = to_number(amplitude)
        self.phase = to_number(phase)
        self.output = output
        self.num_frames = to_number(num_frames)
        self.cur_frame = 0
        self.num_channels = 1

        if str not  in ["sin","cos","complex"]:
            Exception("OscillatorSource only supports sin, cos, or complex")

    def getFrame(self):

        if self.num_frames > 0:
            if self.cur_frame < self.num_frames:
                self.cur_frame += 1
                f = self.getOscFrame()
                return f
            else:
                return None

        return self.getOscFrame()



    def getOscFrame(self):

        dphi = 2.0 * np.pi * self.frequency / self.fs

        phase = self.phase + dphi * np.arange(self.frame_size)

        self.phase = np.mod(phase[-1] + dphi, 2 * np.pi)

        y = None

        if self.output == "complex":
            y= self.amplitude * np.exp(1j * phase)

        elif self.output == "sin":
            y= self.amplitude * np.sin(phase)

        if self.output == "cos":
            y= self.amplitude * np.cos(phase)

        return np.column_stack([y])

    def close(self):
        return

    def get_num_channels(self):
        return self.num_channels

    def summary(self):

        return f"OscillatorSource fs={self.fs} frame_size={self.frame_size}, amplitude={self.amplitude}, num_frames={self.num_frames}, phase={self.phase}, output={self.output}"
