
import importlib
import json
import mydsp
from mydsp.NotchFilter import NotchFilter
from mydsp.LPFilter import LPFilter
from mydsp.UnitFilter import UnitFilter
from mydsp.SincLPFilter import SincLPFilter
from mydsp.HPFilter import HPFilter
from mydsp.BPFilter  import BPFilter
from mydsp.EQFilter import EQFilter
from mydsp.NoiseAddFilter import NoiseAddFilter

from utils.MyLogger import MyLogger, LogLevel
from .dsl_globals import get_context
import matplotlib.pyplot as plt
from .dsl_globals import get_context

all_filters = ["SincLP","LP","BP","EQ","Notch","NoiseAdd","Unit"]

class FilterCommands:

    filters = {}
    def cmd_filter(self, args):
        if len(args) < 2:
            print("Usage: filter <type> <name> [params]")
            return 0

        filter_type, filter_name = args[0], args[1]

        param_str = " ".join(args[2:])
        param_str = param_str.strip("[]")
        class_name = filter_type+"Filter"

        filter_class = globals().get(class_name)

        if not filter_class:
            print(f"Filter class '{class_name}' not found.")
            return 1

        
        params = {}
        params['name']=filter_name
        pairs = [item.split('=') for item in param_str.split(',') if '=' in item]
        for k,v in pairs:
           # params[k] = eval(v, get_context().vars)
            params[k] = get_context().vars.get(v,v)
        MyLogger.log(json.dumps(params, indent=2), LogLevel.INFO)
        if params.get('fs',None) is  None:
            sample_rate = get_context().vars.get('sample_rate',None)
            if sample_rate is None:
                MyLogger.error("cmd_filter: No sample rate defined")
                return 1
            params['fs'] = sample_rate

        if params.get('frame_size',None) is  None:
            frame_size = get_context().vars.get('frame_size',None)
            if frame_size is None:
                MyLogger.error("cmd_filter: No frame_size defined")
                return 1
            params['frame_size'] = frame_size
        print(params)
        f = filter_class(**params)


        self.filters[filter_name] = f
        print(f"Filter '{filter_name}' created.")

        return 0


    def cmd_filters(self, args):
        if not self.filters:
            print("No Filters defined.")
            return 0

        for name in self.filters.keys():
            print(f"- {name}")
        return 0

    def cmd_filtertype(self, args):
        if len(args) != 1:
            print("Usage: filtertype <type>")
            return 1


        class_name = args[0] + "Filter"
        filter_class =   globals().get(class_name)


        if not filter_class:
            print(f"Filter class '{class_name}' not found.")
            return 1

        desc = filter_class.description
        if desc:
            print(f"{class_name}: {desc}")
        else:
            print(f"{class_name} exists but has no description.")

        return 0

    def cmd_list_filters(self,args):

        for ftype in all_filters:
            print(ftype)

        return 0

    def cmd_plot(self,args):

        name = args[0]
        if name not in self.filters:
            print(f"{name} not found")
            return 1

        filter = self.filters[name]
        filter.plotFFT()
        plt.show()
        return 0