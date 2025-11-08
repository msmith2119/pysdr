from commands import dsl_globals
from commands.filter_commands import FilterCommands
from commands.signal_commands import SignalCommands
from commands.pipeline_commands import PipelineCommands
from mydsp import *
class DSLContext(FilterCommands,SignalCommands,PipelineCommands):
    def __init__(self):
        self.vars = {}
        self.filters = {}
        self.signals = {}
        self.pipelines = {}
        self.commands = {
            'set':self.cmd_set,
            'vars':self.cmd_vars,
            'test':self.cmd_test,
            'filter': self.cmd_filter,
            'signal': self.cmd_signal,
            'filters': self.cmd_filters,
            'list_filters':self.cmd_list_filters,
            'signals':self.cmd_signals,
            'pipelines':self.cmd_pipelines,
            'connect':self.cmd_connect,
            'show': self.cmd_show,
            'plot': self.cmd_plot,
            'help': self.cmd_help,
            'filtertype': self.cmd_filtertype,
            'signaltype': self.cmd_signaltype,
        }
        dsl_globals.set_context(self)
    def cmd_test(self,args):
        class_name = "NotchFilter"
        print(globals().keys())
        cl = globals().get(class_name)
        cls = cl.__dict__.get("NotchFilter")
        print(cls)
        description = getattr(cls, "description")
        print(description)
        #print(description)
    def cmd_set(self, args):
        if len(args) != 2:
            print("Usage: set <name> <value>")
            return
        key, value = args
        try:
            value = eval(value, {}, self.vars)  # Evaluate numbers or expressions
        except Exception:
            pass  # Keep it as a string if eval fails
        self.vars[key] = value
        print(f"{key} set to {value}")

    def cmd_vars(self, args):

        for key in self.vars:
            print(f"{key} = {self.vars[key]}")


    def cmd_show(self, args):
        if len(args) != 1:
            print("Usage: show <objectname>")
            return
        name = args[0]
        found = False

        if name  in self.filters:

            print(self.filters[name].summary())
            found = True
        if name in self.signals:

            print(self.signals[name].summary())
            found = True
        if name in self.pipelines:
            src = self.pipelines[name]['src']
            sink = self.pipelines[name]['sink']
            thefilters = self.pipelines[name]['filters']
            print(f"Input src {src.name}")
            print(f"Output sink {sink.name}")
            print(*[f.name for f in thefilters])
            found = True
        if not found:
            print("Object not found")



    def cmd_help(self, args):
        print("Available commands:")
        print("  set var <value> - set a context variable to value")
        print("  vars - list all context variables")
        print("  filter <type> <name> [params] - Create and store a filter")
        print("  signal <type> <name> [params] - Create and store a signal")
        print("  filters                      - List all defined filters")
        print("  plot <name>                  - Plot filter FFT")
        print("  list_filters                 - list all available filter types")
        print("  signals                      - List all defined signals")
        print("  pipelines                    - List all defined pipelines")
        print("  connect <name> <src> (f1 f2 f3...) <sink>")
        print("  show <objectname>           - Show details of a specific object")
        print("  filtertype <type>           - Show parameters for a filter type")
        print("  signaltype <type>           - Show parameters for a signal type")
        print("  help                        - Show this help message")
        print("  exit / quit                 - Exit the REPL")


    def execute_command(self,line):
        parts = line.split()
        cmd, args = parts[0], parts[1:]
        if cmd in self.commands:
            self.commands[cmd](args)
        else:
            print(f"Unknown command: {cmd}. Type 'help' for a list of commands.")

    def run(self):
        print("Custom Filter DSL REPL. Type 'help' for commands. Type 'exit' to quit.")
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


if __name__ == "__main__":
    DSLContext().run()
