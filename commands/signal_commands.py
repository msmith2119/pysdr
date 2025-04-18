from mydsp import Generators, SigClasses
from mydsp.SigClasses import Signal
from .dsl_globals import get_context

class SignalCommands:
    def cmd_signal(self, args):
        if len(args) < 2:
            print("Usage: signal <srctype> <name> [params]")
            return
        src_type, signal_name = args[0], args[1]
        param_str = " ".join(args[2:])

        try:
            params = {}
            params['name'] = signal_name
            if param_str:
                for item in param_str.split():
                    k, v = item.split('=')
                    params[k] = eval(v,get_context().vars)

            if src_type == "sine":
                if 'xl' not in params:
                    params['xl'] = 0.0
                signal = Generators.sineWave(**params)
            elif src_type == "wav":
                signal = Signal(signal_name, 0, 1)
                signal.filepath = params['path']
            else:
                print("unknown src type {src_type")
                return
            self.signals[signal_name] = signal


        except Exception as e:
            print(f"Error creating signal: {e}")

    def cmd_signals(self, args):
        if not self.signals:
            print("No Signals defined.")
            return
        for name in self.signals:
            print(f"- {name}")

    def cmd_signaltype(self, args):
        if len(args) != 1:
            print("Usage: signaltype <type>")
            return
        src_type = args[0]
        description = SigClasses.description(src_type)
        print(description)