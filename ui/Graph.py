
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class Graph(tk.Frame):

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.figure = Figure(figsize=(6, 4))
        self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.line, = self.ax.plot([], [],color='yellow')

        self.set_colors()
    def plot(self, x, y):
        self.line.set_data(x, y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def set_colors(self):

        self.figure.patch.set_facecolor('black')
        self.ax.set_facecolor('black')

        # plotFFT(frame,fs,0,0)
        self.ax.tick_params(colors='yellow')
        self.ax.xaxis.label.set_color('yellow')
        self.ax.yaxis.label.set_color('yellow')
        self.ax.title.set_color('yellow')

        for spine in self.ax.spines.values():
            spine.set_color('yellow')
def test():
    root = tk.Tk()
    root.title("Graph Tester")
    root.geometry("800x500")

    graph = Graph(root)
    graph.pack(fill=tk.BOTH, expand=True)

    # Arbitrary test data
    x = np.linspace(0, 2 * np.pi, 500)
    y = np.sin(x)

    graph.plot(x, y)

    root.mainloop()

if __name__ == "__main__":
    test()