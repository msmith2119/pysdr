
import tkinter as tk
import numpy as np
from ui.EqBand import EqBand

class EqWidget:
    def __init__(self, root,eqfilter, callback):

        self.callback = callback
        bands_frame = tk.Frame(root)
        bands_frame.pack(padx=10, pady=10)

        dbgains = 20*np.log10(eqfilter.gain)
        freqs = eqfilter.fc

        labels = [
        f"{f//1000} kHz" if f % 1000 == 0
        else f"{f/1000:g} kHz" if f >= 1000
        else f"{f} Hz"
        for f in freqs
        ]


        for i in range(len(freqs)):
            band = EqBand(
                bands_frame,
                i,
                labels[i],
                -12,
                12,
                dbgains[i],
                lambda value,idx=i: callback(idx, value)
            )
            band.pack(side=tk.LEFT, padx=6)