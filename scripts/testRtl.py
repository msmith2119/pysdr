

from rtlsdr import RtlSdr

sdr = RtlSdr()

sdr.sample_rate = 250000
sdr.center_freq = 104.5e6
sdr.gain = 'auto'
print("Current Bandwidth (Hz):", sdr.bandwidth)
samples = sdr.read_samples(1024)   # one second of IQ
print(samples[:10])
sdr.close()
