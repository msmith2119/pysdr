

class Decimator:
    def __init__(self,name,factor):
        self.name = name
        self.factor = int(factor)

    description = "decimator <name> [factor=<factor>"

    def doFrame(self,frame):
        if frame is None:
            return None

        return frame[::self.factor]


    def summary(self):
        return f"Decimator {self.name} factor={self.factor}"

    @classmethod
    def from_instance(cls, other):
        return cls(other.name, other.factor)