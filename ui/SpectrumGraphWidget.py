

import tkinter as tk
from ui.SpectrumGraph import SpectrumGraph



class SpectrumGraphWidget(tk.Frame):

    def __init__(self, parent, freq,stop_callback, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.stop_callback = stop_callback
        self.graph = SpectrumGraph(self, freq)
        self.graph.pack(fill=tk.BOTH, expand=True)

        # Future controls can be added here.
        #
        # Example:
        self.control_frame = tk.Frame(self)
        self.control_frame.pack(fill=tk.X)
        #
        tk.Button(
             self.control_frame,
             text="Stop",
             command=self.stop_callback
         ).pack(side=tk.LEFT)
        tk.Button(
            self.control_frame,
            text="Close",
            command=self.close
        ).pack(side=tk.LEFT)
    def update(self, power_db):
        """Update the spectrum display."""
        self.graph.update(power_db)

    def plot(self,power_db):
        self.graph.plot_power(power_db)

    def close(self):
         self.parent.destroy()
def test():
    root = tk.Tk()
    root.title("SpectrumGraphWidget Tester")
    root.geometry("900x500")

    # Example frequency axis
    import numpy as np

    fs = 10000.0
    N = 8192

    freq = np.fft.fftshift(
        np.fft.fftfreq(N, 1 / fs)
    )
    def stopit():
        print("Stop it!")
    widget = SpectrumGraphWidget(root, freq,stopit)
    widget.pack(fill=tk.BOTH, expand=True)

    # Test data
    t = np.arange(N) / fs
    signal = np.sign(np.sin(2 * np.pi * 500 * t))

    spectrum = np.fft.fft(signal)
    power = np.abs(spectrum) ** 2
    power_db = 10 * np.log10(power + 1e-12)
    power_db = np.fft.fftshift(power_db)

   # widget.update(power_db)
    widget.plot(power_db)

    root.mainloop()


if __name__ == "__main__":
    test()

