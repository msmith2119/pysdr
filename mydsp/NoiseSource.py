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
        num_frames=0

    ):
        self.noise_type = noise_type
        self.amplitude = float(amplitude)
        self.frame_size = int(frame_size)
        self.num_channels = int(num_channels)
        self.num_frames = int(num_frames)
        self.current_frame = 0



        self.summary_text = f"Noise Source {self.noise_type} frame_size={self.frame_size} amplitude={self.amplitude} num_channels={self.num_channels} num_frames={self.num_frames}"
    def getFrame(self):
        """
        Returns
        -------
        numpy.ndarray
            Shape (frame_size, channels), dtype=float32
        """




        if self.noise_type == NoiseType.WHITE:
            frame = np.random.uniform(
                low=-self.amplitude,
                high=self.amplitude,
                size=(self.frame_size, self.num_channels)
            )
            return frame.astype(np.float32)

        raise ValueError(f"Unsupported noise type: {self.noise_type}")


    def getMultiFrame(self):

        if  self.num_frames > 0 :
            if self.current_frame < self.num_frames:
              self.current_frame += 1
              return self.getFrame()
            else:
                return None


        return self.getFrame()

    def close(self):
        print("Closing Noise Source")

    def summary(self):
        return self.summary_text