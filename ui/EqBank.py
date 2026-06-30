from ui.EqBand import EqBand

import tkinter as tk

class EqBank:
    def __init__(self,root,freqs,dbgains, callback):
        self.callback = callback

        bands_frame = tk.Frame(root)
        bands_frame.pack(padx=10, pady=10)

        for i in range(len(freqs)):
            band = EqBand(
                bands_frame,
                i,
                f"{freqs[i]}",
                -12,
                12,
                dbgains[i],
                lambda value: callback(i,value)
            )
            band.pack(side=tk.LEFT, padx=6)