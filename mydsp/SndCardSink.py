import sounddevice as sd
import numpy as np

class SndCardSink:
    description = "Sound Card PCM audio sample_rate=fsample,num_channels=num_channels,frame_size=frame_size"
    def __init__(self,
                 sample_rate=48000,
                 num_channels=1,
                 frame_size=1024):

        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.frame_size = frame_size
        self.summary_text = f"Sound Card Sink  frame_size={self.frame_size} sample_rate = {sample_rate} num_channels={num_channels}"
        self.stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=num_channels,
            dtype='float32'
        )

    def start(self):
        self.stream.start()

    def writeFrame(self, frame):
        frame = np.asarray(frame, dtype=np.float32)
        self.stream.write(frame)

    def summary(self):
        return self.summary_text

    def close(self):
        self.stream.stop()
        self.stream.close()