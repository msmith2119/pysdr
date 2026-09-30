

class Decimator:
    def __init__(self,name,frame_size,factor):
        self.name = name
        self.factor = int(factor)
        self.frame_size = frame_size

    description = "decimator <name> [frame_size = <frame_size>,factor=<factor>]"

    def doFrame(self,frame):
        if frame is None:
            return None

        return frame[::self.factor][:self.frame_size]


    def summary(self):
        return f"Decimator {self.name} frame_size = {self.frame_size}, factor={self.factor}"

    @classmethod
    def from_instance(cls, other):
        return cls(other.name, other.frame_size,other.factor)