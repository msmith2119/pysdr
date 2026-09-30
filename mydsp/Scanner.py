
import numpy as np


class Scanner:
    description = "Scanner threshold=<threshold>, decay=<decay>"

    def __init__(self,name,src,meter,meastype,threshold,delay,freqs):
        self.name = name
        self.src = src
        self.meter = meter
        self.meastype = meastype
        self.threshold = float(threshold)
        self.delay = float(delay)
        self.freqs = [float(i) for i in freqs.strip("()[]").split()]

    def summary(self):
        return f"Scanner : name={self.name}, src={self.src}, threshold={self.threshold}, delay={self.delay} freqs={self.freqs}"




