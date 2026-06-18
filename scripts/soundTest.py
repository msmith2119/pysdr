import numpy as np
import sounddevice as sd


def play_tone(freq=440.0,
              duration=2.0,
              sample_rate=48000,
              amplitude=0.25):

    # Time vector
    t = np.arange(int(duration * sample_rate)) / sample_rate

    # Generate sine wave
    samples = amplitude * np.sin(2 * np.pi * freq * t)

    # sounddevice expects float32
    samples = samples.astype(np.float32)

    print(f"Playing {freq} Hz for {duration} seconds...")
    sd.play(samples, samplerate=sample_rate)
    sd.wait()
    print("Done.")


if __name__ == "__main__":
    play_tone(freq=1000.0, duration=3.0)