import re


class PipelineCommands:
    def cmd_connect(self, args):

        line = ' '.join(args)
        expr = r"\s*(\w+)\s+(\w+)\s+\(([^()]+)\)\s+to\s+(\w+)"
        m = re.match(expr, line)

        if not m:
            print("Usage: connect <name> <input_signal> ( <filter1> ... <filterN> ) to <output_signal>")
            return


        try:
            name = m.group(1)
            src = m.group(2)
            filter_names = m.group(3).split()
            sink = m.group(4)

        except (ValueError, IndexError):
            print("Invalid syntax for connect.")
            return

        if src not in self.signals:
            print(f"Input signal '{src}' not found.")
            return
        thefilters =[]
        current_signal = self.signals[src]
        for fname in filter_names:
            filter_obj = self.filters.get(fname)
            if not filter_obj:
                print(f"Filter '{fname}' not found.")
                return
            thefilters.append(filter_obj)
            # Placeholder for actual processing
            #print(f"[{name}] Passing signal '{current_signal.name}' through filter '{fname}'")
            # Example: current_signal = filter_obj.process(current_signal)
        if sink not in self.signals:
            print(f"Output Signal '{sink}' not found")
            return
        pipeline = {}
        pipeline['src'] = current_signal
        pipeline['sink'] = self.signals[sink]
        pipeline['filters'] = thefilters
        self.pipelines[name] = pipeline

        print(f"[{name}] pipeline '{name}' created")


    def cmd_pipelines(self, args):
        if not self.filters:
            print("No Pipelines defined.")
            return
        for name in self.pipelines:
            print(f"- {name}")