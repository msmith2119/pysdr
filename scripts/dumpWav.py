import sys
import numpy as np
from scipy.io import wavfile

def dump_wav_metadata(file_path):
    try:
        # Read the WAV file
        sample_rate, data = wavfile.read(file_path)

        # Determine number of channels
        if len(data.shape) == 1:
            num_channels = 1  # Mono
        else:
            num_channels = data.shape[1]  # Stereo or more

        # Print metadata
        print(f"File: {file_path}")
        print(f"Sample Rate: {sample_rate} Hz")
        print(f"Number of Channels: {num_channels}")
        print(f"Data Type: {data.dtype}")
        print(f"Total Samples: {data.shape[0]}")

        # Print first 10 samples
        print("\nFirst 10 Samples:")
        print(data[:10])

    except Exception as e:
        print(f"Error reading WAV file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dumpWav.py <wav_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    dump_wav_metadata(file_path)
