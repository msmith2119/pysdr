
class ToRealFilter(object):
    def __init__(self,name,fs,frame_size):
        self.name = name
        self.fs = fs
        self.frame_size = frame_size

    description="ToRealFilter. Take real part of IQ"
    def doFrame(self,frame):
        return frame.real

    def summary(self):
        return "ToRealFilter name = {self.name}, fs = {self.fs}, frame_size = {self.frame_size}"

    @classmethod
    def from_instance(cls,other):
        return cls(other.name,other.fs,other.frame_size)

