
class DevNullSink:
    def __init__(self,name):
        self.name = name
    def writeFrame(self,frame):
        return
    def start(self):
        return
    def close(self):
        return
    def summary(self):
        return "DevNullSink"