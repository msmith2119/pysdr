import wave
import numpy as np

class WavFileSource:
    def __init__(self, file_name, frame_size):
        self.file_name = file_name
        self.frame_size = frame_size
        self.wav = wave.open(file_name, 'rb')

        self.num_channels = self.wav.getnchannels()
        self.sample_width = self.wav.getsampwidth()
        self.sample_rate = self.wav.getframerate()
        self.num_frames_total = self.wav.getnframes()

        if self.sample_width != 2:
            raise ValueError("Only 16-bit PCM WAV files are supported.")

    def getFrame(self):
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

        return floats

    def close(self):
        """Close the underlying WAV file."""
        self.wav.close()

# Example usage:
if __name__ == "__main__":
    src = WavFileSource("test.wav", frame_size=1024)
    while True:
        frame = src.getFrame()
        if frame is None:
            break
        print(frame[:10])  # show first 10 samples
    src.close()
