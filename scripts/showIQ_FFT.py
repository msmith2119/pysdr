#!/usr/bin/env python3

import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def main():
    parser = argparse.ArgumentParser(
        description="Plot FFT of IQ data stored as a stereo WAV file."
    )

    parser.add_argument("filename", help="Input WAV file")
    parser.add_argument("offset", type=int,
                        help="Starting sample index")
    parser.add_argument("length", type=int,
                        help="Number of IQ samples")

    args = parser.parse_args()

    # Read WAV file
    fs, data = wavfile.read(args.filename)

    if data.ndim != 2 or data.shape[1] != 2:
        raise RuntimeError("WAV file must contain exactly two channels (I,Q).")

    print(f"Sample Rate : {fs} Hz")
    print(f"Samples     : {len(data)}")
    print(f"Data Type   : {data.dtype}")

    if args.offset + args.length > len(data):
        raise ValueError("Requested segment extends beyond end of file.")

    # Extract segment
    segment = data[args.offset:args.offset + args.length].astype(np.float64)

    # Form complex IQ
    iq = segment[:, 0] + 1j * segment[:, 1]

    # Remove DC
    iq -= np.mean(iq)

    # Apply Hann window
    iq *= np.hanning(len(iq))

    # FFT
    X = np.fft.fftshift(np.fft.fft(iq))
    freq = np.fft.fftshift(np.fft.fftfreq(len(iq), d=1/fs))

    mag = 20 * np.log10(np.abs(X) + 1e-12)

    plt.figure(figsize=(10, 5))
    plt.plot(freq, mag)
    plt.title("IQ Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()