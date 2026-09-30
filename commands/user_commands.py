from mydsp.ScanExecutor import ScanExecutor
from .dsl_globals import get_context


from mydsp.Scanner import Scanner

class UserCommands:


    def cmd_scan(self,args):
        name = args[0]

        param_str = " ".join(args[1:])
        param_str = param_str.strip("[]")

        params = {}
        pairs = [item.split('=') for item in param_str.split(',') if '=' in item]
        for k, v in pairs:
           params[k] = get_context().vars.get(v, v)

        threshold = params.get('threshold',None)
        delay = params.get('delay',None)
        src_name = params.get('src',None)
        meter = params.get('meter',None)
        str_freqs = params.get('freqs',None)

        meastype = params.get('meastype',None)

        if threshold is None:
            threshold = 5.0
        if delay is None:
            delay = 10.0
        if meastype is None:
            meastype="maxpwr"
        missing = False
        if src_name is None:
            print("cmd_scan Error: src is not specified")
            missing = True
        if meter is None:
            print("cmd_scan Error: meter is not specified")
            missing = True
        if missing:
            return 1

        src = self.sources.get(src_name,None)
        if src is None:
            print("cmd_scan Error: src {src_name}  is not defined")
            return 1

        if str_freqs is None:
            print("cmd_scan Error: freqs is not defined")
            return 1

        scanner = Scanner(name,src,meter,meastype,threshold,delay,str_freqs)
        self.scanners[name] = scanner
        print(f"scanner {name} created")
        return 0

    def cmd_start_scan(self,args):

        name = args[0]


        scanner = self.scanners.get(name,None)
        if scanner is None:
            print(f"cmd_start_scan Error: scanner {name}  does not exist")
            return 1



        print("starting scan")

        self.scan_thread = ScanExecutor(scanner,self.pipeline_thread.get_filter_param,self.pipeline_thread.set_filter_param)
        self.scan_thread.start()

        return 0

    def cmd_stop_scan(self,args):

        self.scan_thread.stop()
