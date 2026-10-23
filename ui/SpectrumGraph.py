
import numpy as np
import tkinter as tk
from ui.Graph import Graph
from mydsp.Utils import plotFFT, plot_array
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
class SpectrumGraph(Graph):

    def __init__(self, parent, freq, **kwargs):
        super().__init__(parent, **kwargs)

        self.freq = freq

        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Power (dB)")
        self.smooth = np.zeros(len(freq))
        self.line, = self.ax.plot(freq, np.zeros_like(freq),color='yellow')
        #self.ax.set_ylim(-85, -30)
    def update(self, power):

        smooth = gaussian_filter1d(power, sigma=5)
        if self.smooth is None:
            self.smooth = smooth
        else:
            self.smooth += smooth
            self.smooth *= 0.5

        self.line.set_ydata(self.smooth)


        #self.ax.relim()
        self.canvas.draw_idle()

    def plot_power(self,power):
        self.smooth = gaussian_filter1d(power, sigma=5)
        self.line.set_ydata(self.smooth)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()
def test():
    root = tk.Tk()
    root.title("SpectrumGraph Tester")
    root.geometry("900x500")

    # Sample parameters
    fs = 10000.0
    N = 8192
    square_freq = 500.0

    # Generate a square wave
    t = np.arange(N) / fs
    signal = np.sign(np.sin(2 * np.pi * square_freq * t))


    # FFT
    spectrum = np.fft.fft(signal)

    # Power spectrum
    power = np.abs(spectrum) ** 2

    # Convert to dB
    power_db = 10 * np.log10(power + 1e-12)

    # Put zero frequency in the center
    power_db = np.fft.fftshift(power_db)
    freq = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs))

    # Create graph
    graph = SpectrumGraph(root, freq)
    graph.pack(fill=tk.BOTH, expand=True)

    # Display spectrum
    smooth = gaussian_filter1d(power_db, sigma=5)
    #plot_array(smooth)
    #plt.show()
    graph.plot_power(power_db)

    root.mainloop()


if __name__ == "__main__":
    test()