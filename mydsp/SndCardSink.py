import sounddevice as sd
import numpy as np
import queue
import threading
import time

class SndCardSink:
    description = "Sound Card PCM audio sample_rate=fsample,num_channels=num_channels,frame_size=frame_size"
    def __init__(self,
                 sample_rate,
                 num_channels,
                 frame_size,isComplex=False,useQueue=False):

        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.isComplex = isComplex
        self.useQueue = useQueue
        self.queue_size = 10
        self.start_threshold = self.queue_size/2
        if isComplex:
            self.num_channels = 2
        self.frame_size = frame_size
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.num_channels,
            dtype='float32'
        )

        if self.useQueue:
            self.audio_queue = queue.Queue(maxsize=self.queue_size)
            self.running = True

            self.output_thread = threading.Thread(
                target=self._output_worker,
                daemon=True
            )

    def _output_worker(self):

        while self.audio_queue.qsize() < self.start_threshold:
            time.sleep(0.001)

        while self.running:
            frame = self.audio_queue.get()

            if frame is None:
                self.audio_queue.task_done()
                break

            try:
                self.writeFrameSync(frame)
            finally:
                self.audio_queue.task_done()

    def start(self):
        self.stream.start()
        if self.useQueue:
            self.output_thread.start()



    def writeFrame(self,frame):
        if self.useQueue:
            self.writeFrameAsync(frame)
        else:
            self.writeFrameSync(frame)

    def writeFrameAsync(self,frame):
        self.audio_queue.put(frame)


    def writeFrameSync(self, frame):


        if self.isComplex:
            vals = frame[:, 0]
            frame = np.column_stack([vals.real, vals.imag])
            frame = np.asarray(frame, dtype=np.float32)
        else:
            frame = np.asarray(frame, dtype=np.float32)

        self.stream.write(frame)

    def summary(self):
        return f"Sound Card Sink  frame_size={self.frame_size} sample_rate = {self.sample_rate} num_channels={self.num_channels},useQueue={self.useQueue},complex={self.isComplex}"

    def close(self):

        if self.useQueue:
            self.running = False
            self.audio_queue.put(None)
            self.output_thread.join()
        self.stream.stop()
        self.stream.close()
