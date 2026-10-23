
import numpy as np

from mydsp.Utils import to_bool


class Scanner:
    description = "Scanner threshold=<threshold>, decay=<decay>"

    def __init__(self,name,src,meter,meastype,threshold,delay,freqs,steps):
        self.name = name
        self.src = src
        self.meter = meter
        self.meastype = meastype
        self.threshold = float(threshold)
        self.steps = int(steps)
        self.delay = float(delay)
        self.freqs = [float(i) for i in freqs.strip("()[]").split()]
        self.frange = [float(i) for i in freqs.strip("()[]").split()]

    def summary(self):
        return f"Scanner : name={self.name}, src={self.src}, threshold={self.threshold}, delay={self.delay} freqs={self.freqs},steps={self.steps}"




