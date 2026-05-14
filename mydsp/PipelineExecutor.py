
from utils.MyLogger import MyLogger
from utils.MyLogger import LogLevel

class PipelineExecutor:

    pipelines = None
    src = None
    sink = None
    filters = []

    def __init__(self,pipeline):



        self.pipeline = pipeline
        self.src = pipeline['src']
        self.sink = pipeline['sink']
        self.filters = pipeline['filters']




    def run(self):

        while True:
            frame = self.src.getFrame()
            if frame is None:
                break

            for  f in self.filters:
                frame = f.doFrame(frame)
                if frame is None:
                    break

            if frame is not None:
                self.sink.writeFrame(frame)

        N = len(self.filters)
        for i in range(N):
            print(f"i={i}")
            y = None
            for j in range(i, N):
                print(f"j={j}")
                y = self.filters[j].doFrame(y)
            if y is not None:
                print("writing frame")
                self.sink.writeFrame(y)
        self.sink.close()
        self.src.close()

        return