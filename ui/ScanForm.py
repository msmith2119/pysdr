

import tkinter as tk

from mydsp.Utils import to_number, is_float
from ui import DropDown
from ui.FrequencyControl import FrequencyControl
from ui.DropDown import DropDown
from ui.FrequencyList import FrequencyList
from mydsp.Scanner import Scanner
from functools import partial


class ScanForm(tk.Frame):
    def __init__(self, root,delay,threshold,steps,freqs,run_callback ,stop_callback,**kwargs):
        super().__init__(root, **kwargs)
        self.relief = "solid"
        self.root = root

        tk.Label(
            self,
            text="Freq",
            width=12

        ).grid(row=0, column=0,pady=10)


        def add_freq(freq):


            float_list = [float(i) for i in self.freq_list.get()]

            if freq == 0.0:
                return
            if freq not in float_list:
                self.freq_list.add(freq)



        self.control = FrequencyControl(
            self,
            frequency=0.0

            )
        self.control.grid(row=0, column=1,pady=10)
        tk.Label(
            self,
            text="MHz",
            width=12

        ).grid(row=0, column=2)

        apply_button = tk.Button(
            self,
            text="Add",
            command=lambda: add_freq(self.control.getFrequency())
        )
        apply_button.grid(row=0, column=3,pady=10)

        tk.Label(
            self,
            text="Scan",
            width=12

        ).grid(row=1, column=0, pady=10)


        self.freq_list = FrequencyList(
            self,
            freqs

        )

        self.freq_list.grid(row=1, column=1,pady=10,padx=0)
        tk.Label(
            self,
            text="MHz",
            width=12

        ).grid(row=1, column=2)

        def remove_last():
            self.freq_list.pop()

        del_button = tk.Button(
            self,
            text="Delete",
            command=lambda: remove_last()
        )
        del_button.grid(row=1, column=3, pady=10)
        def doscan():
            self.exec_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            run_callback(self.entry_delay.get(), self.entry_threshold.get(),self.entry_steps.get(),self.freq_list.get())
        self.exec_button = tk.Button(
            self,
            text="Run",
            command=lambda: doscan()
        )
        self.exec_button.grid(row=3, column=3, pady=10)
        tk.Label(
            self,
            text="Delay",
            width=12

        ).grid(row=3, column=0, pady=10)
        self.entry_delay = tk.Entry(
           self,
           width=12
        )
        self.entry_delay.grid(row=3, column=1, pady=10)
        self.entry_delay.insert(0,f"{delay:.1f}")
        def stopscan():
            self.stop_button.config(state=tk.DISABLED)
            self.exec_button.config(state=tk.NORMAL)
            stop_callback()
        tk.Label(
            self,
            text="sec",
            width=12

        ).grid(row=3, column=2,pady=10)
        self.stop_button = tk.Button(
            self,
            text="Stop",
            command=lambda: stopscan()
        )
        self.stop_button.grid(row=4, column=3, pady=10)
        self.stop_button.config(state=tk.DISABLED)
        tk.Label(
            self,
            text="Threshold",
            width=12

        ).grid(row=4, column=0, pady=10)

        self.entry_threshold = tk.Entry(
            self,
            width=12
        )
        self.entry_threshold.grid(row=4, column=1, pady=10)
        self.entry_threshold.insert(0,f"{threshold:.1f}")
        tk.Label(
            self,
            text="mW",
            width=12

        ).grid(row=4, column=2, pady=10)
        self.stop_button = tk.Button(
            self,
            text="Stop",
            command=lambda: stopscan()
        )
        self.stop_button.grid(row=4, column=3, pady=10)
        self.stop_button.config(state=tk.DISABLED)
        tk.Label(
            self,
            text="Steps",
            width=12

        ).grid(row=5, column=0, pady=10)
        self.entry_steps = tk.Entry(
            self,
            width=12
        )
        self.entry_steps.grid(row=5, column=1, pady=10)
        self.entry_steps.insert(0, f"{steps}")


        self.close_button = tk.Button(
            self,
            text="Close",
            command=root.destroy
        )
        self.close_button.grid(row=6, column=3, pady=10)
if __name__ == "__main__":

    def doscan(name,delay,threshold,steps,freqs):
        print(name)
        print(delay)
        print(threshold)
        print(steps)
        print(freqs)
    def stopscan():
        print("stopscan")

    root = tk.Tk()
    root.title("Scan Control")
    delay = 9.0
    threshold = 5.0
    steps = 0
    freqs = [120.9,135.1]
    str_freqs = ["120.9","135.1"]
    form = ScanForm(root,delay,threshold,steps,str_freqs,partial(doscan,"myscan"),stopscan)


    form.pack(padx=20, pady=20)

    root.mainloop()