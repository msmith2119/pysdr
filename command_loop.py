import os

import numpy as np

import tkinter as tk
from functools import partial
from commands import dsl_globals
from commands.filter_commands import FilterCommands
from commands.signal_commands import SignalCommands
from commands.pipeline_commands import PipelineCommands
from commands.wav_commands import WavCommands
from commands.io_commands import IOCommands
from mydsp import *
from mydsp.MorseCodeSource import MorseCodeSource
from mydsp.Utils import parse_argv, plot_array, create_ola_function, to_number
from ui.SliderControl import SliderControl
from utils.MyLogger import MyLogger
from utils.MyLogger import LogLevel
import sys
import re
import matplotlib.pyplot as plt

MyLogger.set_level(LogLevel.INFO)
class DSLContext(FilterCommands,SignalCommands,IOCommands,PipelineCommands,WavCommands):
    def __init__(self):
        self.vars = {}
        self.filters = {}
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
            'signal': self.cmd_signal,
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
            'signals':self.cmd_signals,
            'pipelines':self.cmd_pipelines,
            'run':self.cmd_run_pipeline,
            'stop':self.cmd_stop_pipeline,
            'connect':self.cmd_connect,
            'exec':self.cmd_exec,
            'show': self.cmd_show,
            'plot': self.cmd_plot,
            'help': self.cmd_help,
            'quit':self.cmd_quit,
            'filtertype': self.cmd_filtertype,
            'signaltype': self.cmd_signaltype,
        }
        dsl_globals.set_context(self)

    def cmd_test(self,args):

        name = args[0]

        if self.pipeline_thread is None:
            MyLogger.error("No pipeline running")
            return



        filter = self.filters[name]
        params = filter.parameters()

        def value_changed(param,value):
            print(f"set {name} {param} val {value}")
            self.pipeline_thread.set_filter_param(name, param, value)

        root = tk.Tk()
        root.title(f"Filter {name}")
        for param in params:
            getter = getattr(filter, f"{param}_range")
            rng = getter()
            cval = self.pipeline_thread.get_filter_param(name, param)
            SliderControl(
                root,
                param,
                rng[0],
                rng[1],
                cval,
                partial(value_changed,param)
            )
        tk.Button(
            root,
            text="Close",
            command=root.destroy
        ).pack(pady=10)

        root.mainloop()
        return


    def cmd_widget_param(self,args):

      
        name = args[0]

        if self.pipeline_thread is None:
            MyLogger.error("No pipeline running")
            return

        filter = self.filters[name]
        params = filter.parameters()

        def value_changed(param, value):
            print(f"set {name} {param} val {value}")
            self.pipeline_thread.set_filter_param(name, param, value)

        root = tk.Tk()
        root.title(f"Filter {name}")
        for param in params:
            getter = getattr(filter, f"{param}_range")
            rng = getter()
            cval = self.pipeline_thread.get_filter_param(name, param)
            SliderControl(
                root,
                param,
                rng[0],
                rng[1],
                cval,
                partial(value_changed, param)
            )
        tk.Button(
            root,
            text="Close",
            command=root.destroy
        ).pack(pady=10)

        root.mainloop()
        return

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

        print("set filter param {parts[0} {parts[1]}")
        self.pipeline_thread.set_filter_param(parts[0],parts[1],value)



    def cmd_show(self, args):
        if len(args) != 1:
            print("Usage: show <objectname>")
            return
        name = args[0]
        found = False

        if name  in self.filters:
            print(self.filters[name].summary())
            found = True
        if name in self.sources:
            print(self.sources[name].summary())
            found = True
        if name in self.sinks:
            print(self.sinks[name].summary())
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
        if param_value == None:
            MyLogger.log(f"Unknown parameter {param_name}",LogLevel.WARN)
            return None


        return param_value;


    def set_dev_param(self,path,value):

        a = self.get_dev_obj_param(path)
        if a is None:
            return


        setter = getattr(a[0], f"set_{a[1]}")
        print(f"value = {value}")
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
