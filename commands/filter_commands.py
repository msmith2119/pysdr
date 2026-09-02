
import importlib
import json
import mydsp
from mydsp.Decimator import Decimator
from mydsp.NotchFilter import NotchFilter
from mydsp.LPFilter import LPFilter
from mydsp.UnitFilter import UnitFilter
from mydsp.DelayFilter import DelayFilter
from mydsp.HPFilter import HPFilter
from mydsp.BPFilter  import BPFilter
from mydsp.EQFilter import EQFilter
from mydsp.NoiseAddFilter import NoiseAddFilter
from mydsp.AnalogFilter import AnalogFilter
from mydsp.RCFilter import RCFilter
from mydsp.ToneControlFilter import ToneControlFilter
from mydsp.SincFilter import SincFilter
from mydsp.FMModFilter import FMModFilter
from mydsp.FMDemodFilter import FMDemodFilter
from mydsp.AMDemodFilter import AMDemodFilter
from mydsp.FreqShiftFilter import FreqShiftFilter
from mydsp.FMStereoFilter import FMStereoFilter
from mydsp.Utils import to_number
from utils.MyLogger import MyLogger, LogLevel
from .dsl_globals import get_context
import matplotlib.pyplot as plt
from .dsl_globals import get_context

all_filters = ["SincLP","LP","BP","EQ","Notch","Delay","NoiseAdd","Analog","RC","ToneControl","Sinc","FreqShift","FMDemod","AMDemod","FmStereo","Unit"]

class FilterCommands:

    filters = {}
    decimators = {}
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
        f = filter_class(**params)


        self.filters[filter_name] = f
        print(f"Filter '{filter_name}' created.")

        return 0

    def cmd_decimator(self,args):

        param_str = " ".join(args[1:])
        param_str = param_str.strip("[]")
        name = args[0]
        params = {}
        params['name'] = name
        pairs = [item.split('=') for item in param_str.split(',') if '=' in item]
        for k, v in pairs:
            params[k] = v
        dec = Decimator(**params)
        self.decimators[name] = dec
        fs = get_context().vars.get('sample_rate',None)
        if fs is None:
            print("sample_rate not defined")
        frame_size = get_context().vars.get('frame_size',None)
        if frame_size is None:
            print("frame_size not defined")

        if fs is None  or frame_size is None:
            print("Decimator : Unable to reset global parameters ")
            return 0
        factor = params.get('factor',None)
        if factor is None:
            print("Decimator : factor not defined")
            return 0
        fs_new = int(fs)/int(factor)
        frame_size_new = int(int(frame_size)/int(factor))
        get_context().vars['sample_rate']=fs_new
        get_context().vars['frame_size']=frame_size_new
        print(f"Decimator {name} created.")


    def cmd_filters(self, args):
        if not self.filters and not self.decimators:
            print("No Filters defined.")
            return 0

        for name in  [*self.filters.keys(), *self.decimators.keys()]:
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
        fa = 1.0

        if name not in self.filters:
            print(f"{name} not found")
            return 1
        params = {}
        if len(args)  > 1:
            param_str = " ".join(args[1:])
            pairs = [item.split('=') for item in param_str.split(',') if '=' in item]
            for k,v in pairs:
                params[k] = to_number(v)


        fa = params.get('max',1.0)
        filter = self.filters[name]
        filter.plotFFT(fa)
        #filter.plot_impulse()
        plt.show()
        return 0
