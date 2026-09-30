import threading
import numpy as np
import time

from utils.MyLogger import MyLogger
from utils.MyLogger import LogLevel

class ScanExecutor(threading.Thread):

    def __init__(self,scanner,param_getter,param_setter):
        super().__init__()
        self.scanner = scanner
        self.running = False
        self.param_getter = param_getter
        self.param_setter = param_setter

    def run(self):


        freqs =  self.scanner.freqs

        self.running = True

        while self.running:
            for freq in freqs:
                print(f"Listening  to {freq}")
                active = True
                self.scanner.src.sdr.center_freq = freq * 1e6
                while active:
                    self.param_setter(self.scanner.meter,self.scanner.meastype,0.0)
                    time.sleep(self.scanner.delay)
                    if self.running == False:
                        return
                    pwr = self.param_getter(self.scanner.meter, self.scanner.meastype)
                    print(f"{self.scanner.meastype} = {pwr}")

                    if  pwr < self.scanner.threshold:
                        active = False


    def stop(self):
        print("stopping scan")
        self.running = False