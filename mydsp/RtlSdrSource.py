
from rtlsdr import RtlSdr

from mydsp.Utils import to_number
import numpy as np
import queue
import threading

class RtlSdrSource:
    Params = ["sample_rate","freq","frame_size"]
    def __init__(self,name,sample_rate,freq, frame_size,num_frames=0):
        self.name = name
        self.frame_size = to_number(frame_size)
        self.num_frames = to_number(num_frames)
        self.sdr = RtlSdr()
        self.num_channels = 1
        self.sample_rate = int(sample_rate)
        self.sdr.sample_rate = int(sample_rate)
        self.sdr.center_freq = int(float(freq))
        self.sdr.gain = 'auto'
        self.current_frame = 0

        self.queue = queue.Queue(maxsize=10)
        self.running = True

        self.thread = threading.Thread(
            target=self._capture,
            daemon=True
        )

    def _capture(self):
        while self.running:
            frame = self.getInternalFrame()
            self.queue.put(frame)

    def getFrame(self):
        return self.queue.get()

    def getInternalFrame(self):


        if self.num_frames  == 0:
            return self.sdr.read_samples(self.frame_size)

        if self.current_frame < self.num_frames:
            self.current_frame += 1
            samples = self.sdr.read_samples(self.frame_size)
            return np.column_stack([samples])
        return None

    def get_num_channels(self):
        return self.num_channels

    def summary(self):
        return f"RtlSdr  Source frame_size={self.frame_size}, freq={self.sdr.center_freq}, num_frames={self.num_frames} "

    def start(self):
        self.thread.start()

    def close(self):
        self.running = False
        self.thread.join()
        self.sdr.close()




