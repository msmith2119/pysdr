import numpy as np
import matplotlib.pyplot as plt

def sinc_bandpass(f1, f2, fs, N, window=np.hamming):
    # Normalize frequencies to [0, 0.5]
    f1_norm = f1 / fs
    f2_norm = f2 / fs

    n = np.arange(N)
    mid = (N - 1) / 2

    # Ideal impulse responses (sinc) for LPFs
    h1 = 2 * f1_norm * np.sinc(2 * f1_norm * (n - mid))
    h2 = 2 * f2_norm * np.sinc(2 * f2_norm * (n - mid))

    # BPF = LPF(f2) - LPF(f1)
    h = h2 - h1

    # Apply window
    h *= window(N)

    return h

# === Parameters ===
fs = 8000           # Sampling rate
f1, f2 = 1000, 2000 # Band edges in Hz
N = 101             # Filter length (odd is best)

bpf = sinc_bandpass(f1, f2, fs, N)

# === Plot ===
w, H = np.fft.rfftfreq(len(bpf), 1/fs), np.abs(np.fft.rfft(bpf))

plt.figure(figsize=(10, 4))
plt.plot(w, 20 * np.log10(H + 1e-6))  # dB scale
plt.title(f"FIR Band-Pass Filter: {f1}–{f2} Hz")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.tight_layout()
plt.show()
