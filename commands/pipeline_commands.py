import re
from logging import Logger

from mydsp.PipelineExecutor import PipelineExecutor
from utils import MyLogger


class PipelineCommands:
    def cmd_connect(self, args):

        line = ' '.join(args)
        expr = r"\s*(\w+)\s+(\w+)\s+\(([^()]+)\)\s+to\s+(\w+)"
        m = re.match(expr, line)

        if not m:
            print("Usage: connect <name> <input_signal> ( <filter1> ... <filterN> ) to <output_signal>")
            return 1


        try:
            name = m.group(1)
            src = m.group(2)
            filter_names = re.split(r'[\s,]+',m.group(3))
            sink = m.group(4)

            print(filter_names)

        except (ValueError, IndexError):
            print("Invalid syntax for connect.")
            return 1

        if src not in self.sources:
            print(f"Input signal '{src}' not found.")
            return 1
        thefilters =[]
        current_source = self.sources[src]
        for fname in filter_names:
            filter_obj = self.filters.get(fname)
            if not filter_obj:
                print(f"Filter '{fname}' not found.")
                return 1
            thefilters.append(filter_obj)
            # Placeholder for actual processing
            #print(f"[{name}] Passing signal '{current_signal.name}' through filter '{fname}'")
            # Example: current_signal = filter_obj.process(current_signal)
        if sink not in self.sinks:
            print(f"Output Signal '{sink}' not found")
            return 1
        pipeline = {}
        pipeline['src'] = self.sources[src]
        pipeline['sink'] = self.sinks[sink]
        pipeline['filters'] = thefilters
        self.pipelines[name] = pipeline

        print(f"[{name}] pipeline '{name}' created")

        return 0


    def cmd_run_pipeline(self, args):

        line = ' '.join(args)
        print(f"line = {line}")
        expr = r"\s*(\w+)"
        m = re.match(expr, line)

        if not m:
            print("Usage: run pipeline")
            return 1

        pname = m.group(1)
        if pname not in self.pipelines:
            print(f"Pipeline '{pname}' not found.")
            return 1
        pipe_line = self.pipelines[pname]
        src = self.pipelines[pname]['src']
        sink = self.pipelines[pname]['sink']
        thefilters = self.pipelines[pname]['filters']


        print(f"Running pipeline : {pname}")

        pipe_executor = PipelineExecutor(pipe_line)
        pipe_executor.run()


    def cmd_pipelines(self, args):
        if not self.filters:
            print("No Pipelines defined.")
            return 1
        for name in self.pipelines:
            print(f"- {name}")

        return 0