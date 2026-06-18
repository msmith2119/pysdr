import re
import os
from pathlib import Path

from mydsp.MorseCodeSource import MorseCodeSource
from mydsp.NoiseSource import NoiseSource, NoiseType
from mydsp.SndCardSink import SndCardSink
from mydsp.WavFileSink import WavFileSink
from mydsp.Utils import is_writable
from utils.MyLogger import MyLogger
from utils.MyLogger import LogLevel

all_sources = ["WavFile"]
all_sinks = ["WavFile"]

from mydsp.WavFileSource import WavFileSource
from .dsl_globals import get_context
from mydsp.Utils import to_number

noise_params = ['amplitude']
morse_params  = ['msg_file','amplitude','wpm','tone']
sndcard_params = ['sample_rate','num_channels','frame_size']
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
        expr = r"([\w_]+)\s+([\w_]+)\s+\[([^\[\]]*)\]\s*"
        #print(f"line = {line}")
        m = re.match(expr, line)

        if not m:
            print("Usage: source <type> <name> [path=path_to_file,frame_size=frame_size]")
            return 1

        stype = m.group(1)
        name = m.group(2)
        params = m.group(3)

        if stype == "WavFile":
            return self.gen_wavesource(name,params)
        elif stype == "Noise":
            return self.gen_noisesource(name,params)
        elif stype == "Morse":
            return self.gen_morsesource(name,params)
        else :
            MyLogger.error(f"Unknown source type: {stype}")
            return 1

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
        expr = r"([\w_]+)\s+([\w_]+)\s+\[([^\[\]]*)\]\s*"

        m = re.match(expr, line)

        if not m:
            print("Usage: sink <type> <name> [path=path_to_file,sample_rate=rate,num_channels=n,frame_size=frame_size]")
            return 1


        stype = m.group(1)
        name = m.group(2)
        params = m.group(3)

        sink = None

        if stype == "WavFile":
            sink = self.gen_wav_sink(name,params)
        elif stype == "SndCard":
            sink = self.gen_sndcard_sink(name,params)
        else:
            MyLogger.error(f"Invalid sink type {stype}")
            return 1

        if sink is None:
            return 1

        self.sinks[name] = sink
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


    def gen_wavesource(self,name,param_str):


        items = param_str.split(',')
        params = {}
        for item in items:
            k, v = item.split('=')
            params[k] = get_context().vars.get(v, v)

        if 'path' not in params:
            print("path parameter missing")
            return 1

        path = params['path']
        if not Path(path).exists():
            print(f"audio source {path} not found")
            return 1

        frame_size = int(params.get('frame_size',1))
        if 'frame_size' not in params:
            frame_size = get_context().vars.get('frame_size',None)
            if frame_size is None:
                print("frame_size parameter  not defined")
                return 1




        src = WavFileSource(path, frame_size)
        get_context().vars['sample_rate'] = src.sample_rate
        self.sources[name] = src
        print(f"Source {name} created")
        return 0

    def gen_noisesource(self,name,param_str):


        items = param_str.split(',')
        params = {}
        for item in items:
            k, v = item.split('=')
            params[k] = get_context().vars.get(v, v)


        for pname in noise_params:


            if params.get(pname,None ) is None:
                MyLogger.log(f'WavCommands.cmd_gen_noise: {pname}  not found ', LogLevel.ERROR)
                return

        num_frames = 0
        if params.get("num_frames",None) is not None:
            num_frames = int(params['num_frames'])

        frame_size = int(params.get('frame_size', 1))
        if 'frame_size' not in params:
            frame_size = get_context().vars.get('frame_size', None)
            if frame_size is None:
                print("cmd_noise_source: frame_size parameter  not defined")
                return 1

        num_channels = int(params.get('num_channels', 1))
        if 'num_channels' not in params:
            num_channels = get_context().vars.get('num_channels', None)
            if num_channels is None:
                print("cmd_noise_source: num_channels parameter  not defined")
                return 1

        amplitude = float(params['amplitude'])

        src  = NoiseSource(NoiseType.WHITE,amplitude,frame_size,num_channels,num_frames)

        self.sources[name] = src
        print(f"Source {name} created")
        return 0

    def gen_morsesource(self,name,param_str):

        items = param_str.split(',')
        params = {}
        for item in items:
            k, v = item.split('=')
            params[k] = get_context().vars.get(v, v)

        param_values = {}
        for pname in morse_params:

            param_values[pname] = params.get(pname, None)
            if param_values[pname] is None:
                MyLogger.log(f'IOCommands.cmd_gen_morsesource: {pname}  not found ', LogLevel.ERROR)
                return
            param_values[pname] = to_number(param_values[pname])

        frame_size = int(params.get('frame_size', 1))
        if 'frame_size' not in params:
            frame_size = get_context().vars.get('frame_size', None)
            if frame_size is None:
                print("cmd_morse_source: frame_size parameter  not defined")
                return 1

        sample_rate = int(params.get('sample_rate', 1))
        if 'sample_rate' not in params:
            sample_rate = get_context().vars.get('sample_rate', None)
            if sample_rate is None:
                print("cmd_morse_source: sample_rate parameter  not defined")
                return 1


        amplitude = float(params['amplitude'])
        msg_file = params['msg_file']
        wpm = float(params['wpm'])
        tone = float(params['tone'])
        frame_size = max(1, frame_size)

        if not Path(msg_file).exists():
            print(f"audio source {msg_file} not found")
            return 1

        msrc = MorseCodeSource(sample_rate, frame_size, msg_file, amplitude,wpm, tone)
        self.sources[name] = msrc
        print(f"Source {name} created")
        return 0

    def gen_wav_sink(self,name,param_str):


        items = param_str.split(',')
        params = {}
        for item in items:
            k, v = item.split('=')
            params[k] = get_context().vars.get(v, v)
            val = self.get_dev_param(v)
            if val is not None:
                params[k] = val
        if 'path' not in params:
            print("path parameter missing")
            return None
        path = params['path']
        if not is_writable(path):
            print(f"Cannot write to path {path}")
            return None

        sample_rate = int(params.get('sample_rate', 1))
        if 'sample_rate' not in params:
            sample_rate = get_context().vars.get('sample_rate', None)
            if sample_rate is None:
                print("sample_rate parameter  not defined")
                return 1

        num_channels = int(params.get('num_channels', 1))
        if 'num_channels' not in params:
            num_channels = get_context().vars.get('num_channels', None)
            if num_channels is None:
                print("num_channels parameter  not defined")
                return 1

        frame_size = int(params.get('frame_size', 1))
        if 'frame_size' not in params:
            frame_size = get_context().vars.get('frame_size', None)
            if frame_size is None:
                print("frame_size parameter  not defined")
                return 1


        sink = WavFileSink(path, num_channels, frame_size, sample_rate)

        return sink

    def gen_sndcard_sink(self,name,param_str):

        items = param_str.split(',')
        params = {}
        for item in items:
            if "=" not in item:
                continue
            k, v = item.split('=')
            params[k] = get_context().vars.get(v, v)


        sample_rate = int(params.get('sample_rate', 1))
        if 'sample_rate' not in params:
            sample_rate = get_context().vars.get('sample_rate', None)
            if sample_rate is None:
                print("gen_sndcard_sink : sample_rate parameter  not defined")
                return 1

        num_channels =  int(params.get('num_channels', 1))
        if 'num_channels' not in params:
            num_channels = get_context().vars.get('num_channels', None)
            if num_channels is None:
                print("num_channels parameter  not defined")
                return 1

        frame_size =  int(params.get('frame_size', 1))
        if 'frame_size' not in params:
            frame_size = get_context().vars.get('frame_size', None)
            if frame_size is None:
                print("frame_size  parameter  not defined")
                return 1


        sink = SndCardSink(sample_rate,num_channels,frame_size)

        return sink

