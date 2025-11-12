import wave
import numpy as np

class WavFileSink:
    def __init__(self, file_name, frame_size, sample_rate=48000):
        """
        Create a WAV file sink that writes float frames (-1..1) as 16-bit PCM.

        Args:
            file_name: Output WAV file path.
            frame_size: Number of samples per frame (for buffering).
            sample_rate: Sampling rate in Hz (default 48kHz).
        """
        self.file_name = file_name
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.num_channels = 1  # mono for now

        # Open the WAV file for writing
        self.wav = wave.open(file_name, 'wb')
        self.wav.setnchannels(self.num_channels)
        self.wav.setsampwidth(2)  # 16-bit
        self.wav.setframerate(sample_rate)

    def writeFrame(self, frame):
        """Write one frame (NumPy array of floats in range -1..1) to file."""
        frame = np.asarray(frame, dtype=np.float32)

        # Clip to avoid overflow if DSP chain exceeds ±1
        np.clip(frame, -1.0, 1.0, out=frame)

        # Convert float → int16 PCM
        int_samples = np.round(frame * 32767.0).astype(np.int16)

        # Write to file as raw bytes
        self.wav.writeframes(int_samples.tobytes())

    def close(self):
        """Close the WAV file."""
        self.wav.close()

# Example usage:
if __name__ == "__main__":
    # Generate a 1 kHz test tone for 1 second
    sr = 48000
    duration = 1.0
    t = np.arange(0, duration, 1/sr)
    signal = 0.5 * np.sin(2 * np.pi * 1000 * t)

    frame_size = 1024
    sink = WavFileSink("test_out.wav", frame_size, sample_rate=sr)

    # Write in frames
    for i in range(0, len(signal), frame_size):
        frame = signal[i:i+frame_size]
        sink.writeFrame(frame)
    sink.close()
    print("Wrote test_out.wav")
