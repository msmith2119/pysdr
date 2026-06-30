import threading

import numpy as np

from utils.MyLogger import MyLogger
from utils.MyLogger import LogLevel

class PipelineExecutor(threading.Thread):

    pipelines = None
    src = None
    sink = None
    filter_banks = []

    def __init__(self,pipeline):
        super().__init__()
        self.running = False
        self.pipeline = pipeline
        self.src = pipeline['src']
        self.sink = pipeline['sink']
        self.filters = pipeline['filters'] 

        self.channels = self.src.num_channels
        self.frame_size = self.src.frame_size


        for i in range(self.channels):
            filts = []
            filts.append(self.filters[0].from_instance(self.filters[0]))
            for j in range(1,len(self.filters)):
                filts.append(self.filters[j].from_instance(self.filters[j]))
            self.filter_banks.append(filts)




    def set_filter_param(self,fname,pname,pvalue):

        for row in self.filter_banks:
            for filt in row:
                if filt.name  == fname:
                    setter = getattr(filt, f"set_{pname}")
                    setter(pvalue)

    def get_filter_param(self,fname,pname):
        for row in self.filter_banks:
            for filt in row:
                if filt.name == fname:
                    value =getattr(filt, pname)
                    return value

    def run(self):

        self.running = True
        self.sink.start()
        while self.running:
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



        if self.running:
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

    def stop(self):
        self.running = False


    def dump_filters(self):

        for row in self.filter_banks:
            for filt in row:
                print(filt.summary())
