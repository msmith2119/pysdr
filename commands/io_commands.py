import re
import os
from pathlib import Path

from mydsp.WavFileSink import WavFileSink
from mydsp.Utils import is_writable
from utils.MyLogger import MyLogger
from utils.MyLogger import LogLevel

all_sources = ["WavFile"]
all_sinks = ["WavFile"]

from mydsp.WavFileSource import WavFileSource
from .dsl_globals import get_context

class IOCommands:


    def cmd_exec(self,args):

        if len(args) == 0:
            print("Usage : exec <filename>")
            return 1

        fname = args[0]

        if not os.path.isfile(fname) and  not os.access(fname, os.R_OK):
            print(f"File {fname} does not exist or is not readable")
            return 1



        get_context().runFile(fname)

        return 0

    def cmd_input_src(self,args):

        line = ' '.join(args)
        expr = r"([\w_]+)\s+([\w_]+)\s+\[([^\[\]]+)\]\s*"
        #print(f"line = {line}")
        m = re.match(expr, line)

        if not m:
            print("Usage: source <type> <name> [path=path_to_file,frame_size=frame_size]")
            return 1

        stype = m.group(1)
        name = m.group(2)
        params = m.group(3)

        items = params.split(',')
        params = {}
        for item in items:
            k, v = item.split('=')
            params[k] = get_context().vars.get(v,v)

        if 'path' not in params:
            print("path parameter missing")
            return 1

        path = params['path']
        if not Path(path).exists():
            print("audio source {path} not found")
            return 1

        if 'frame_size' not in params:
            print("frame_size parameter missing")
            return 1

        frame_size = int(params['frame_size'])
        frame_size= max(1, frame_size)

        src = WavFileSource(path,frame_size)
        get_context().vars['sample_rate'] = src.sample_rate
        self.sources[name]=src
        print(f"Source {name} created")
        return 0

    def cmd_sourcetype(self, args):
        if len(args) != 1:
            print("Usage: sourcetype <type>")
            return 1
        source_name = args[0]
        class_name = source_name + "Source"
        MyLogger.log(f"classname = {class_name}",LogLevel.INFO)
        source_class = globals().get(class_name)
        desc = source_class.description
        print(f"{source_name}:{desc}")
        return 0

    def cmd_sources(self, args):
        if not self.sources:
            print("No Sources defined.")
            return 1
        for name in self.sources:
            print(f"- {name}")

        return 0

    def cmd_list_sources(self,args):

        for stype in all_sources:
            print(stype)

        return 0

    def cmd_output_sink(self,args):
        line = ' '.join(args)
        expr = r"([\w_]+)\s+([\w_]+)\s+\[([^\[\]]+)\]\s*"

        m = re.match(expr, line)

        if not m:
            print("Usage: sink <type> <name> [path=path_to_file,sample_rate=rate,frame_size=frame_size]")
            return 1

        stype = m.group(1)
        name = m.group(2)
        params = m.group(3)

        items = params.split(',')
        params = {}
        for item in items:
            k, v = item.split('=')
            params[k] = get_context().vars.get(v,v)

        if 'path' not in params:
            print("path parameter missing")
            return 1
        path = params['path']
        if not is_writable(path):
            print(f"Cannot write to path {path}")
            return 1

        if 'sample_rate' not in params:
            print("sample_rate parameter missing")
            return 1
        sample_rate = int(params['sample_rate'])
        if 'frame_size' not in params:
            print("frame_size parameter missing")
            return 1

        frame_size = int(params['frame_size'])
        frame_size= max(1, frame_size)

        sink = WavFileSink(path,frame_size,sample_rate)
        self.sinks[name]=sink
        print(f"Sink {name} created")
        return 0

    def cmd_sinktype(self, args):
        if len(args) != 1:
            print("Usage: sinktype <type>")
            return 1

        sink_name = args[0]
        class_name = sink_name + "Source"
        MyLogger.log(f"classname = {class_name}", LogLevel.INFO)
        sink_class = globals().get(class_name)
        desc = sink_class.description
        print(f"{sink_name}:{desc}")

    def cmd_sinks(self, args):
        if not self.sinks:
            print("No Sinks defined.")
            return 1
        for name in self.sinks:
            print(f"- {name}")

        return 0

    def cmd_list_sinks(self, args):

        for stype in all_sinks:
            print(stype)

        return 0