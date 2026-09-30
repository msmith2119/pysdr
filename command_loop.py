import os

import tkinter as tk
from functools import partial

from jupyterlab.commands import enable_extension

from commands import dsl_globals
from commands.dsl_globals import get_context
from commands.filter_commands import FilterCommands
from commands.user_commands import UserCommands
from commands.pipeline_commands import PipelineCommands
from commands.wav_commands import WavCommands
from commands.io_commands import IOCommands
from mydsp import WavFileSource, Utils
from mydsp.LPFilter import LPFilter
from mydsp.NoiseSource import NoiseSource
from mydsp.OscillatorSource import OscillatorSource
from mydsp.FreqShiftFilter import FreqShiftFilter
from mydsp.MeterFilter import MeterFilter
from mydsp.SincFilter import SincFilter
from mydsp.SineWaveSource import SineWaveSource
from mydsp.EQFilter import EQFilter
from mydsp.RtlFileSource import RtlFileSource
from matplotlib import pyplot as plt
from ui.EqBand import EqBand
from mydsp.WavFileSource import WavFileSource
from mydsp.Scanner import Scanner
from mydsp.ScanExecutor import ScanExecutor
from ui.EqWidget import EqWidget
from ui.ParamWidget import  ParamWidget
from ui.RtlSdrForm import RtlSdrForm
from ui.ScanForm import ScanForm
from mydsp.Utils import parse_argv, plot_array, plotFFT, to_number,shift_freq,plot_arrays
from ui.SliderControl import SliderControl
from utils.MyLogger import MyLogger
from utils.MyLogger import LogLevel
from mydsp.Utils import is_float
import sys
import math
import numpy as np
from scipy.fftpack import fft, ifft
import numpy as np
from scipy.ndimage import gaussian_filter1d
import time
from scipy.fftpack import fft, ifft
import heapq

MyLogger.set_level(LogLevel.INFO)
class DSLContext(FilterCommands,IOCommands,PipelineCommands,WavCommands,UserCommands):
    def __init__(self):
        self.vars = {}
        self.filters = {}
        self.scanners = {}
        self.signals = {}
        self.pipelines = {}
        self.sources = {}
        self.sinks = {}
        self.pipeline_thread = None
        #self.root = tk.Tk()
        #self.root.withdraw()
        self.commands = {
            'set':self.cmd_set,
            'vars':self.cmd_vars,
            'test':self.cmd_test,
            'filter': self.cmd_filter,
            'decimator':self.cmd_decimator,
            'source': self.cmd_input_src,
            'sources':self.cmd_sources,
            'sourcetype':self.cmd_sourcetype,
            'sink': self.cmd_output_sink,
            'sinks': self.cmd_sinks,
            'sinktype': self.cmd_sinktype,
            'addwaves': self.cmd_addwaves,
            'gennoise': self.cmd_gennoise,
            'list_sinks':self.cmd_list_sinks,
            'filters': self.cmd_filters,
            'filter_types':self.cmd_list_filters,
            'list_sources':self.cmd_list_sources,
            'set_dev_param':self.cmd_set_dev_param,
            'widget':self.cmd_widget_param,
            'set_pipeline_param':self.cmd_set_pipeline_param,
            'get_profile':self.cmd_get_pipeline_profile,
            'pipelines':self.cmd_pipelines,
            'run':self.cmd_run_pipeline,
            'stop':self.cmd_stop_pipeline,
            'scan':self.cmd_scan,
            'start_scan':self.cmd_start_scan,
            'stop_scan':self.cmd_stop_scan,
            'connect':self.cmd_connect,
            'exec':self.cmd_exec,
            'show': self.cmd_show,
            'plot': self.cmd_plot,
            'help': self.cmd_help,
            'quit':self.cmd_quit,
            'filtertype': self.cmd_filtertype

        }
        dsl_globals.set_context(self)



    def cmd_test(self,args):

        #src = self.sources["mywav"]
        frames = []
        frame_size = 5000
        fs = 48000.0
        meter = MeterFilter("mymeter",fs,frame_size)
        #src = NoiseSource("myn",0.1,frame_size,1,num_frames=30)
        src = WavFileSource("mywav","audio/atc_noise.wav",frame_size)
        while True:
            block = src.getFrame()
            if block is None:
                break
            frames.append(block[:,0])


        for frame in frames[1:2]:
            #plotFFT(frame,fs,0,0)
            #plt.show()
            plot_array(frame)
            plt.show()
            fft_frame = fft(frame)
            power = np.abs(fft_frame) ** 2

            power = np.fft.fftshift(power)
            power = np.asarray(power, dtype=float)
            top = heapq.nlargest(5,power)
            print(top)
            plot_array(power)
            plt.show()
            spectrum_db = 10.0 * np.log10(power)
            smooth_db = gaussian_filter1d(spectrum_db, 2)
            noise_db = gaussian_filter1d(spectrum_db, 20)
            snr_db = smooth_db - noise_db
            plot_array(snr_db)
            plt.show()
            meter.doFrame(frame)
            rms = meter.get_rms()
            x = frame- np.mean(frame)

            #plot_array(x)

            # Envelope
            env = np.abs(x)

            # Smooth envelope
            #filt = LPFilter("mylp",src.sample_rate,1000.0,0.001,src.frame_size,1.0,False)
            filt = SincFilter("mysinc",fs,src.frame_size,101,1000,0,"LOWPASS")
            env = filt.doFrame(env)
            #plot_array(env)
            #plt.show()
            # Autocorrelation
            r = np.correlate(env, env, mode='full')

            mid = len(env) - 1
            r /= r[mid]
            #plot_array(r)
            #plt.show()
            # Ignore zero lag
            h = int(0.75*len(r))
            #score = np.max(r[mid + 1:mid + 200])
            score = r[h]
            print(f"rms = {rms}")
            print(f"score={score}")




    def show_scan_widget(self,scanner):


        def doscan(name,delay,threshold,str_freqs):

            freqs = [float(i)  for i in str_freqs]
            print(name)
            print(delay)
            print(threshold)
            print(freqs)
            if not is_float(delay):
                MyLogger.error(f"Not a number for delay {delay}")
                return
            if not is_float(threshold):
                MyLogger.error(f"Not a number for threshold {threshold}")
                return
            scanner = self.scanners.get(name,None)
            if scanner is None:
                MyLogger.error("referenced scanner not defined {name}")
                return
            scanner.delay = float(delay)
            if scanner.delay <= 0.0:
                MyLogger.error(f"Non positive number found for delay {scanner.delay} ")
                return
            scanner.threshold = float(threshold)
            if scanner.threshold <= 0.0:
                MyLogger.error(f"Non positive number found for threshold {scanner.threshold}")
                return
            if len(freqs) < 2:
                MyLogger.error("Need two or more frequencies to scan")
                return

            scanner.freqs = freqs
            self.scan_thread = ScanExecutor(scanner, self.pipeline_thread.get_filter_param,
                                            self.pipeline_thread.set_filter_param)
            self.scan_thread.start()
        def stopscan():
            self.scan_thread.stop()
            print("stopscan")
            return
        print(scanner.freqs)
        root = tk.Tk()
        root.title(f"Scanner")
        form = ScanForm(root,scanner.delay,scanner.threshold,scanner.freqs,partial(doscan,scanner.name),stopscan)
        form.pack(padx=20, pady=20)
        root.mainloop()

    def cmd_widget_param(self,args):

        name = args[0]

      #  if self.pipeline_thread is None:
      #      MyLogger.error("No pipeline running")
      #      return

        filter = self.filters.get(name,None)
        src = self.sources.get(name,None)
        scanner = self.scanners.get(name,None)
        if filter is not None:
            if type(filter).__name__ == "EQFilter":
                self.show_eq_widget(filter)
            else:
                self.show_filter_widget(name)
        elif src is not None:
            if type(src).__name__ == "RtlSdrSource":
                rtlsdr = src.sdr
                self.show_rtlsdr_widget(rtlsdr)
            else:
                MyLogger.info("No widget support for source {name}")
        elif scanner is not None:
            self.show_scan_widget(scanner)
        else:
            MyLogger.error(f"Object not found {name}")

    def show_rtlsdr_widget(self,rtlsdr):

        freq = str(rtlsdr.center_freq/1e6)
        gain = rtlsdr.gain


        vals = {}
        vals['frequency'] = freq
        vals['gain'] = gain

        root = tk.Tk()
        root.title(f"RtlSdr")
        def frequency_changed(frequency):
            freq = float(frequency)*1e6
            rtlsdr.center_freq = freq
            print(f"Frequency changed : {freq}")

        def gain_changed(gain):
            if gain == "auto":
                rtlsdr.gain='auto'
            else:
                rtlsdr.gain=float(gain)
                print(f"Gain changed : {gain}")

        form = RtlSdrForm(root, vals, frequency_changed, gain_changed)
        form.pack(padx=20, pady=20)
        root.mainloop()

    def show_filter_widget(self,name):



        if self.pipeline_thread is None:
            MyLogger.error("No pipeline running")
            return

        N = 100
        filter = self.filters[name]
        def reset_params():
            orig_params = filter.getParameters()
            for p in orig_params:
                self.pipeline_thread.set_filter_param(name, p.name, p.val)
        params = filter.getParameters()
        for param in params:
            param.val = self.pipeline_thread.get_filter_param(name, param.name)

        def param_changed(pname, value):
            self.pipeline_thread.set_filter_param(name,pname,value)

        root = tk.Tk()
        root.title(f"Filter {name}")
        param_editor = ParamWidget(
            root,name,params,N,param_changed
        )
        param_editor.pack(padx=20, pady=20)


        root.mainloop()
        return




    def show_eq_widget(self,eqfilter):

        root = tk.Tk()
        root.title(f"EqBand")
        name = eqfilter.name
        pname = "dbgain"
        def eqchange(i, val):
            value = f"{i}:{val}"
            self.pipeline_thread.set_filter_param(name, pname, value)

        eqwidget = EqWidget(root, eqfilter, eqchange)

        root.mainloop()

    def cmd_set(self, args):
        if len(args) != 2:
            print("Usage: set <name> <value>")
            return
        key, value = args
        try:
            value = eval(value, {}, self.vars)  # Evaluate numbers or expressions
        except Exception:
            pass  # Keep it as a string if eval fails
        val = self.get_dev_param(value)
        if val is not None:
            self.vars[key] = val
        else:
            self.vars[key] = value
        print(f"{key} set to {value}")

    def cmd_vars(self, args):

        for key in self.vars:
            print(f"{key} = {self.vars[key]}")

    def cmd_set_dev_param(self,args):

        path = args[0]
        value = " ".join(args[1:])

        self.set_dev_param(path,value)



    def cmd_set_pipeline_param(self,args):

        path = args[0]
        value = args[1]

        parts = path.split('.')

        if len(parts) != 2:
            # MyLogger.log(f"Invalid path {path}",LogLevel.INFO)
            return None

        names = {
            'inst': parts[0],
            'pname': parts[1],

        }

        self.pipeline_thread.set_filter_param(parts[0],parts[1],value)


    def cmd_get_pipeline_profile(self,args):
        name = args[0]
        data = self.pipeline_thread.get_filter_profile(name)
        print(np.mean(data))
        total = np.mean(self.pipeline_thread.profile_data)
        print(f"Total pipleline delay = {total}")

    def cmd_show(self, args):
        if len(args) != 1:
            print("Usage: show <objectname>")
            return
        name = args[0]
        found = False

        if name  in self.filters:
            print(self.filters[name].summary())
            found = True
        if name in self.decimators:
            print(self.decimators[name].summary())
        if name in self.sources:
            print(self.sources[name].summary())
            found = True
        if name in self.sinks:
            print(self.sinks[name].summary())
            found=True
        if name in self.scanners:
            print(self.scanners[name].summary())
            found=True
        if name in self.pipelines:
            src = self.pipelines[name]['src']
            sink = self.pipelines[name]['sink']
            thefilters = self.pipelines[name]['filters']

            print(f"Input src {src.summary()}")
            print(f"Output sink {sink.summary()}")
            #print(*[f.name for f in thefilters])

            n = 1
            for f in thefilters:
                print(f"filter {n}: {f.summary()}")
                n+=1
            found = True
        if not found:
            print("Object not found")


    def get_dev_obj_param(self,path):



        parts = path.split('.')

        if len(parts) != 3:
            # MyLogger.log(f"Invalid path {path}",LogLevel.INFO)
            return None

        names = {
            'type': parts[0],
            'name': parts[1],
            'param': parts[2]
        }

        inst_name = names['name']
        param_name = names['param']
        obj = None

        if names['type'] == "filter":
            obj = self.filters.get(inst_name)
            if obj == None:
                MyLogger.log(f"Invalid filter name {inst_name}", LogLevel.WARN)
                return None
        elif names['type'] == "source":
            obj = self.sources.get(inst_name)
            if obj == None:
                MyLogger.log(f"Invalid source name {inst_name}", LogLevel.WARN)
                return None
        elif names['type'] == "sink":
            obj = self.sinks.get(inst_name)
            if obj == None:
                MyLogger.log(f"Invalid sink name {inst_name}", LogLevel.WARN)
                return None

        else:
            MyLogger.log(f"Invalid type {names['type']}", LogLevel.WARN)
            return None

        return [obj,param_name]

    def get_dev_param(self,path):

        if not isinstance(path, str):
            return None

        a = self.get_dev_obj_param(path)
        if a is  None:
            return None
        param_name = a[1]
        param_value = getattr(a[0], a[1],None)

        if param_value is not None:
            return param_value


        getter = getattr(a[0], f"get_{a[1]}")
        if getter is None:
            MyLogger.log(f"Unknown parameter {param_name}", LogLevel.WARN)

        param_value = getter()

        return param_value;


    def set_dev_param(self,path,value):

        a = self.get_dev_obj_param(path)
        if a is None:
            return


        setter = getattr(a[0], f"set_{a[1]}")
        setter(value)

    def object_type(self, object_type,sub_type):


        class_name = sub_type + object_type
        cl = globals().get(class_name)
        cls = cl.__dict__.get(class_name)



        if not cls:
            print(f"{object_type} class '{class_name}' not found.")
            return

        desc = cls.description
        if desc:
            print(f"{class_name}: {desc}")
        else:
            print(f"{class_name} exists but has no description.")

    def cmd_help(self, args):
        print("Available commands:")
        print("  set var <value> - set a context variable to value")
        print("  vars - list all context variables")
        print("  filter <type> <name> [params] - Create and store a filter")
        print("  signal <type> <name> [params] - Create and store a signal")
        print("  source <type> <name> [params] - Create and store a source")
        print("  filters                      - List all defined filters")
        print("  plot <name>                  - Plot filter FFT")
        print("  list_filters                 - list all available filter types")
        print("  list_sources                 - list all available source types")
        print("  signals                      - List all defined signals")
        print("  exec filename               - execute the commands in filename")
        print("  pipelines                    - List all defined pipelines")
        print("  run pipeline                - run the pipeline")
        print("  connect <name> <src> (f1 f2 f3...) <sink>")
        print("  show <objectname>           - Show details of a specific object")
        print("  filtertype <type>           - Show parameters for a filter type")
        print("  signaltype <type>           - Show parameters for a signal type")
        print("  sourcetype <type>           - Show parameters for an source type")
        print("  help                        - Show this help message")
        print("  exit / quit                 - Exit the REPL")

    def cmd_quit(self,args):
        exit()
    def execute_command(self,line):
        parts = line.split()
        cmd, args = parts[0], parts[1:]

        print(f"** {line}")
        if cmd in self.commands:
            return self.commands[cmd](args)
        else:
            print(f"Unknown command: {cmd}. Type 'help' for a list of commands.")
            return 1

    def run(self):
        print("Custom Filter DSL REPL. Type 'help' for commands. Type 'exit' to quit.")

        opts = parse_argv(sys.argv)
        if "f" in opts:
            if os.path.isfile(opts["f"]):
                filename = opts["f"]
                self.runFile(filename)
                exit(0)
        while True:
            try:
                line = input(">> ").strip()
                if not line:
                    continue
                if line in ('exit', 'quit'):
                    print("Goodbye.")
                    break
                self.execute_command(line)

            except KeyboardInterrupt:
                print("\n(Use 'exit' to quit)")
            except Exception as e:
                print(f"Error: {e}")

    def runFile(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith('#'):
                    continue  # ignore empty lines or comments
                if self.execute_command(line) == 1 :
                    print(f"Error halting exec({filename})")
                    break


if __name__ == "__main__":
    DSLContext().run()
