import numpy as np


class NullSource:
    def __init__(self,name,frame_size,num_channels ):

        self.name = name
        self.frame_size = frame_size
        self.num_channels = num_channels

        self.summary_text = f"NullSource  frame_size={self.frame_size}  num_channels={self.num_channels}"
        self.vals = np.zeros((frame_size,num_channels),dtype=float)


    def getMultiFrame(self):
        return self.vals

    def close(self):
            return

    def summary(self):
            return self.summary_text