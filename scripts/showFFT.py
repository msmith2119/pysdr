

#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def main():
    parser = argparse.ArgumentParser(
        description="Plot FFT of a segment of a WAV file."
    )
    parser.add_argument("filename", help="Input WAV file")
    parser.add_argument("offset", type=int,
                        help="Starting sample index")
    parser.add_argument("length", type=int,
                        help="Number of samples to analyze")

    args = parser.parse_args()

    # Read WAV
    fs, data = wavfile.read(args.filename)

    print(f"Sample rate : {fs} Hz")
    print(f"Shape       : {data.shape}")
    print(f"Dtype       : {data.dtype}")

    # If stereo, use first channel
    if data.ndim > 1:
        print("Stereo file detected. Using channel 0.")
        data = data[:, 0]

    if args.offset + args.length > len(data):
        raise ValueError("Requested segment extends beyond end of file.")

    # Convert to float
    samples = data[args.offset:args.offset + args.length].astype(np.float64)
    x = samples.astype(np.float32) / 32768.0
    print(x[:100])
    # Remove DC
    x -= np.mean(x)

    # Optional window
    x *= np.hanning(len(x))

    # FFT
    X = np.fft.rfft(x)
    freq = np.fft.rfftfreq(len(x), d=1/fs)

    mag = 20 * np.log10(np.abs(X) + 1e-12)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(freq, mag)
    plt.title("FFT Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()