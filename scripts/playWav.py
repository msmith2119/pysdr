

#!/usr/bin/env python3

import argparse

import sounddevice as sd
from scipy.io import wavfile


def main():
    parser = argparse.ArgumentParser(description="Play a WAV file.")
    parser.add_argument("filename", help="WAV file to play")
    args = parser.parse_args()

    # Read WAV file
    sample_rate, data = wavfile.read(args.filename)

    print(f"File       : {args.filename}")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Shape      : {data.shape}")
    print(f"DType      : {data.dtype}")

    # Play and wait for completion
    sd.play(data, sample_rate)
    sd.wait()


if __name__ == "__main__":
    main()