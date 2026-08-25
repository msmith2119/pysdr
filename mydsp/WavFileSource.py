import wave
import numpy as np

from mydsp.Utils import to_bool

class WavFileSource:
    Params = ["path","frame_size"]
    description = "Wave File PCM audio source path=filepath, frame_size=frame_size"
    def __init__(self, name,path, frame_size,loop="False",complex="False"):
        self.name = name
        self.file_name = path
        self.frame_size = frame_size
        self.wav = wave.open(self.file_name, 'rb')
        self.loop = to_bool(loop)
        self.isComplex = to_bool(complex)
        self.num_channels = self.wav.getnchannels()
        self.sample_width = self.wav.getsampwidth()
        self.sample_rate = self.wav.getframerate()
        self.num_frames_total = self.wav.getnframes()
        self.summary_text = f"Wave File Source path={self.file_name} frame_size={self.frame_size} sample_rate={self.sample_rate} num_channels={self.num_channels}, isComplex={self.isComplex}, loop={self.loop}"
        if self.sample_width != 2:
            raise ValueError("Only 16-bit PCM WAV files are supported.")

    def getMonoFrame(self):
        """Read the next frame_size samples as float32 in range (-1, 1)."""
        raw_bytes = self.wav.readframes(self.frame_size)
        if not raw_bytes:
            return None  # End of file

        # Convert byte data to numpy int16 array
        samples = np.frombuffer(raw_bytes, dtype=np.int16)

        # Handle stereo: take only the first channel
        if self.num_channels > 1:
            samples = samples[::self.num_channels]

        # Normalize to range (-1, 1)
        floats = samples.astype(np.float32) / 32768.0
        if len(floats) < self.frame_size:
            floats = np.pad(floats, (0, self.frame_size - len(floats)), mode="constant")
        return floats

    def getMultiFrame(self):
        """
        Return next block of audio samples as float32 matrix.

        Shape:
            (block_size, num_channels)

        Note:
            WAV 'frame' = one multi-channel sample
            block_size = number of WAV frames returned per call
        """

       # raw_bytes = self.wav.readframes(self.frame_size)


        #if not raw_bytes:
         #   return None


        raw_bytes = self.getRawFrame()

        if raw_bytes is None:
            return None

        samples = np.frombuffer(raw_bytes, dtype=np.int16)

        frames_read = len(samples) // self.num_channels

        # reshape into (frames, channels)
        samples = samples.reshape(frames_read, self.num_channels)

        floats = samples.astype(np.float32) / 32768.0

        # pad final block if short
        if frames_read < self.frame_size:
            pad_rows = self.frame_size - frames_read

            padding = np.zeros(
                (pad_rows, self.num_channels),
                dtype=np.float32
            )

            floats = np.vstack((floats, padding))

        return floats

    def getFrame(self):

        frame = self.getMultiFrame()


        if  self.isComplex:
            return self.toComplexFrame(frame)

        return  frame

    def toComplexFrame(self,frame):

        if frame  is None:
            return frame

        real = frame[:, 0]
        imag = frame[:, 1]

        col = real.astype(np.complex64) + 1j * imag.astype(np.complex64)
        cols = [(col)]

        return np.column_stack(cols)



    def start(self):
        return
    def close(self):
        self.wav.close()

    def summary(self):
        return self.summary_text

    def get_num_channels(self):
        if self.isComplex:
            return 1
        return self.num_channels

    def summary(self):
        return self.summary_text

    def getRawFrame(self):

        framesNeeded = self.frame_size
        chunks = []

        while framesNeeded > 0:

            raw = self.wav.readframes(framesNeeded)

            if len(raw) == 0:

                if not self.loop:

                    if len(chunks) == 0:
                        return None

                    break

                self.wav.rewind()
                continue

            framesRead = len(raw) // (self.num_channels * self.sample_width)

            chunks.append(raw)
            framesNeeded -= framesRead

        raw = b"".join(chunks)

        # convert raw -> float ndarray
        return raw


# Example usage:
if __name__ == "__main__":
    src = WavFileSource("test.wav", frame_size=1024)
    while True:
        frame = src.getFrame()
        if frame is None:
            break
        print(frame[:10])  # show first 10 samples
    src.close()
