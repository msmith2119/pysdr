
import wave

import numpy as np
import sounddevice as sd




wav = wave.open("../audio/cqi2.wav", "rb")
sample_rate = wav.getframerate()
num_channels = wav.getnchannels()
print(f"sample_rate={sample_rate}")
raw_bytes = wav.readframes(wav.getnframes())
samples = np.frombuffer(raw_bytes, dtype=np.int16)
if num_channels > 1:
    samples = samples[::num_channels]
audio = samples.astype(np.float32) / 32768.0
wav.close()
sd.play(audio, samplerate=sample_rate)
sd.wait()
#print(sd.query_devices())



