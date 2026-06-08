import numpy as np

from utils.MyLogger import MyLogger
from utils.MyLogger import LogLevel

class PipelineExecutor:

    pipelines = None
    src = None
    sink = None
    filter_banks = []

    def __init__(self,pipeline):



        self.pipeline = pipeline
        self.src = pipeline['src']
        self.sink = pipeline['sink']
        self.filters = pipeline['filters'] 

        self.channels = self.src.num_channels
        self.frame_size = self.src.frame_size
        print(f"channels = {self.channels}")

        for i in range(self.channels):
            filts = []
            filts.append(self.filters[0].from_instance(self.filters[0]))
            for j in range(1,len(self.filters)):
                filts.append(self.filters[j].from_instance(self.filters[j]))
            self.filter_banks.append(filts)





    def run2(self):


        print(f"filters_banks[0] = {self.filter_banks[0]}")
        while True:
            block = self.src.getMultiFrame()
            cols = []
            if block is None:
                break
            for i in range(self.channels):
                frame = block[:,i]
                for j in range(len(self.filters)):

                    frame = self.filter_banks[i][j].doFrame(frame)
                    if frame is None:
                        break
                if frame is not None:
                    cols.append(frame)


            if len(cols) > 0 :
                self.sink.writeFrame(np.column_stack(cols))



        for k in range(len(self.filters)):

            colvec = []
            for i in range(self.channels):
                y = None
                for j in range(i,len(self.filters)):
                    y = self.filter_banks[i][j].doFrame(y)
                if y is not None:
                    colvec.append(y)

            if len(colvec) > 0 :
                self.sink.writeFrame(np.column_stack(colvec))



        self.src.close()
        self.sink.close()

    def run(self):

        while True:
            frame = self.src.getMonoFrame()
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
            y = None
            for j in range(i, N):
                y = self.filters[j].doFrame(y)
            if y is not None:
                self.sink.writeFrame(y)
        self.sink.close()
        self.src.close()

        return