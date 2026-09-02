

import tkinter as tk

from mydsp.Utils import to_number
from ui import DropDown
from ui.FrequencyControl import FrequencyControl
from ui.DropDown import DropDown


class RtlSdrForm(tk.Frame):
    def __init__(self, root,vals,freq_callback,gain_callback ,**kwargs):
        super().__init__(root, **kwargs)
        self.relief = "solid"
        self.vals = vals
        self.root = root
        self.freq_callback = freq_callback
        self.gain_callback = gain_callback

        freq = to_number(vals['frequency'])


        #self.pack(padx=20, pady=20)

        tk.Label(
            self,
            text="Freq",
            width=12

        ).grid(row=0, column=0,pady=10)



        control = FrequencyControl(
            self,
            frequency=freq

            )
        control.grid(row=0, column=1,pady=10)
        tk.Label(
            self,
            text="MHz",
            width=12

        ).grid(row=0, column=2)

        apply_button = tk.Button(
            self,
            text="Apply",
            command=lambda: freq_callback(control.getFrequency())
        )
        apply_button.grid(row=0, column=3,pady=10)

        tk.Label(
            self,
            text="Gain",
            width=12
        ).grid(row=1,column=0,pady=10)

        gain_values = ["auto", "0.0", "10.0", "20.0", "30.0", "40.0"]
        gain_select = DropDown(self,label="Gain",initial="auto",values=gain_values,callback=gain_callback)
        gain_select.grid(row=1, column=1)

        tk.Label(
            self,
            text="dB",
            width=12
        ).grid(row=1,column=2,pady=10)

if __name__ == "__main__":

    def frequency_changed(frequency):
        print(f"Frequency: {frequency:.3f} MHz")

    def gain_chained(gain):
        print(f"gain = {gain}")
    root = tk.Tk()
    root.title("Frequency Control")
    vals = {}
    vals['frequency'] = "104.5"
    form = RtlSdrForm(root,vals,frequency_changed,gain_chained)


    form.pack(padx=20, pady=20)

    root.mainloop()