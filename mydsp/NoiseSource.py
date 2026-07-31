from enum import Enum
import numpy as np


class NoiseType(Enum):
    WHITE = 1


class NoiseSource:
    """
    Generates frames of noise samples.

    Parameters
    ----------
    noise_type : NoiseType
        Type of noise to generate.
    amplitude : float
        Peak amplitude of the noise. Samples will lie in
        [-amplitude, +amplitude].
    frame_size : int
        Number of samples returned by getFrame().
    channels : int
        Number of channels in the output frame.
    """
    description = "source Noise <name> frame_size=<frame_size>,num_channels = <num_channels>, amplitude=<amplitude>"
    def __init__(
        self,
        noise_type: NoiseType,
        amplitude: float,
        frame_size: int,
        num_channels: int,
        num_frames=0,
        isComplex=False

    ):
        self.noise_type = noise_type
        self.amplitude = float(amplitude)
        self.frame_size = int(frame_size)
        self.num_channels = int(num_channels)
        self.num_frames = num_frames
        self.cur_frame = 0
        self.isComplex = isComplex
        if self.isComplex:
            self.num_channels = 1


        self.summary_text = f"Noise Source {self.noise_type} frame_size={self.frame_size} amplitude={self.amplitude} num_channels={self.num_channels} "


    def getFrame(self):

        if self.num_frames > 0:
            if self.cur_frame < self.num_frames:
                self.cur_frame += 1
                f = self.getNoiseFrame()
                if self.isComplex:
                    return self.toComplexFrame(f)
                return f
            else:
                return None

        frame = self.getNoiseFrame()
        if self.isComplex:
            return self.toComplexFrame(frame)
        return frame


    def getNoiseFrame(self):
        """
        Returns
        -------
        numpy.ndarray
            Shape (frame_size, channels), dtype=float32
        """

        num_channels = self.num_channels
        if self.isComplex:
            num_channels  = 2
        if self.noise_type == NoiseType.WHITE:
            frame = np.random.uniform(
                low=-self.amplitude,
                high=self.amplitude,
                size=(self.frame_size, num_channels)
            )

            return frame.astype(np.float32)

        raise ValueError(f"Unsupported noise type: {self.noise_type}")



    def toComplexFrame(self,frame):


        real = frame[:, 0]
        imag = frame[:, 1]

        col = real.astype(np.complex64) + 1j * imag.astype(np.complex64)
        cols = [(col)]

        return np.column_stack(cols)

    def close(self):
        return

    def summary(self):
        return self.summary_text