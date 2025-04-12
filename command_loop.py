from mydsp import Generators, SigClasses
from mydsp.NotchFilter import NotchFilter
from mydsp.SigClasses import Signal, description


class DSLContext:
    def __init__(self):
        self.vars = {}
        self.filters = {}
        self.signals = {}
        self.commands = {
            'set':self.cmd_set,
            'vars':self.cmd_vars,
            'filter': self.cmd_filter,
            'signal': self.cmd_signal,
            'filters': self.cmd_filters,
            'signals':self.cmd_signals,
            'show': self.cmd_show,
            'help': self.cmd_help,
            'filtertype': self.cmd_filtertype,
            'signaltype': self.cmd_signaltype,
        }

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
        for key, value in self.vars.items():
            print(f"{key} = {value}")


    def cmd_filter(self, args):
        if len(args) < 2:
            print("Usage: filter <type> <name> [params]")
            return
        filter_type, filter_name = args[0], args[1]
        param_str = " ".join(args[2:])

        try:
            params = {}
            params['name']=filter_name
            if param_str:
                for item in param_str.split():
                    k, v = item.split('=')
                    params[k] = eval(v, {}, self.vars)

            filter_class = globals().get(filter_type.capitalize() + "Filter")
            if not filter_class:
                print(f"Unknown filter type: {filter_type}")
                return

            f = filter_class(**params)
            self.filters[filter_name] = f
            print(f"Filter '{filter_name}' created.")

        except Exception as e:
            print(f"Error creating filter: {e}")

    def cmd_signal(self, args):
        if len(args) < 2:
            print("Usage: signal <srctype> <name> [params]")
            return
        src_type, signal_name = args[0], args[1]
        param_str = " ".join(args[2:])

        try:
            params = {}
            params['name']=signal_name
            if param_str:
                for item in param_str.split():
                    k, v = item.split('=')
                    params[k] = eval(v, {}, self.vars)
            signal = None
            if src_type == "sine":
                signal =Generators.sineWave(**params)
            elif src_type == "wav":
                signal = Signal.from_file(params['path'])
            else:
                print("unknown src type {src_type")
                return
            self.signals[signal_name] = signal
            print(f"Signal '{signal_name}' created.")

        except Exception as e:
            print(f"Error creating signal: {e}")

    def cmd_filters(self, args):
        if not self.filters:
            print("No filters defined.")
            return
        for name in self.filters:
            print(f"- {name}")

    def cmd_signals(self, args):
        if not self.signals:
            print("No Signals defined.")
            return
        for name in self.signals:
            print(f"- {name}")

    def cmd_show(self, args):
        if len(args) != 1:
            print("Usage: show <filtername>")
            return
        name = args[0]
        found = False
        if name  in self.filters:
            print(self.filters[name].summary())
            found = True
        if name in self.signals:
            print(self.signals[name].summary())
            found = True
        if not found:
            print("Object not found")

    def cmd_filtertype(self, args):
        if len(args) != 1:
            print("Usage: filtertype <type>")
            return
        class_name = args[0].capitalize() + "Filter"
        filter_class = globals().get(class_name)
        if not filter_class:
            print(f"Filter class '{class_name}' not found.")
            return
        desc = getattr(filter_class, "description", None)
        if desc:
            print(f"{class_name}: {desc}")
        else:
            print(f"{class_name} exists but has no description.")

    def cmd_signaltype(self, args):
        if len(args) != 1:
            print("Usage: signaltype <type>")
            return
        src_type = args[0]
        description = SigClasses.description(src_type)
        print(description)


    def cmd_help(self, args):
        print("Available commands:")
        print("  set var <value> - set a context variable to value")
        print("  vars - list all context variables")
        print("  filter <type> <name> [params] - Create and store a filter")
        print("  signal <type> <name> [params] - Create and store a signal")
        print("  filters                      - List all defined filters")
        print("  signals                      - List all defined signals")
        print("  show <objectname>           - Show details of a specific object")
        print("  filtertype <type>           - Show parameters for a filter type")
        print("  signaltype <type>           - Show parameters for a signal type")
        print("  help                        - Show this help message")
        print("  exit / quit                 - Exit the REPL")

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

                parts = line.split()
                cmd, args = parts[0], parts[1:]
                if cmd in self.commands:
                    self.commands[cmd](args)
                else:
                    print(f"Unknown command: {cmd}. Type 'help' for a list of commands.")
            except KeyboardInterrupt:
                print("\n(Use 'exit' to quit)")
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    DSLContext().run()
