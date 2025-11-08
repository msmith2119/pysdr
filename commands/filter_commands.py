
import importlib

import mydsp
from mydsp.NotchFilter import NotchFilter
from mydsp.LPFilter import LPFilter
from mydsp.SincLPFilter import SincLPFilter
from .dsl_globals import get_context
import matplotlib.pyplot as plt
from .dsl_globals import get_context

all_filters = ["SincLP","LP","Notch"]

class FilterCommands:

    def cmd_filter(self, args):
        if len(args) < 2:
            print("Usage: filter <type> <name> [params]")
            return

        filter_type, filter_name = args[0], args[1]

        param_str = " ".join(args[2:])
        param_str = param_str.strip("[]")
        class_name = filter_type+"Filter"

        filter_class = globals().get(class_name)

        if not filter_class:
            print(f"Filter class '{class_name}' not found.")
            return

        params = {}
        params['name']=filter_name
        pairs = [item.split('=') for item in param_str.split(',') if '=' in item]
        for k,v in pairs:
            params[k] = eval(v, get_context().vars)

        f = filter_class(**params)


        self.filters[filter_name] = f
        print(f"Filter '{filter_name}' created.")



    def cmd_filters(self, args):
        if not self.filters:
            print("No Filters defined.")
            return
        for name in self.filters:
            print(f"- {name}")

    def cmd_filtertype(self, args):
        if len(args) != 1:
            print("Usage: filtertype <type>")
            return


        class_name = args[0] + "Filter"


        filter_class =   globals().get(class_name)


        if not filter_class:
            print(f"Filter class '{class_name}' not found.")
            return

        desc = filter_class.description
        if desc:
            print(f"{class_name}: {desc}")
        else:
            print(f"{class_name} exists but has no description.")

    def cmd_list_filters(self,args):

        for ftype in all_filters:
            print(ftype)

    def cmd_plot(self,args):

        name = args[0]
        if name not in self.filters:
            print(f"{name} not found")
            return

        filter = self.filters[name]
        filter.plotFFT()
        plt.show()
