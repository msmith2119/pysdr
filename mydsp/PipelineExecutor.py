
class PipelineExecutor:

    src = None
    sink = None
    filters = []

    def __init__(self,src,sink,filters):
        self.src = src
        self.sink = sink
        self.filters=filters

    def run(self):
        return