
import tkinter as tk
from tkinter import ttk


class DropDown(tk.Frame):

    def __init__(self, parent, label, values, initial=None, callback=None):
        super().__init__(parent)

        self.callback = callback



        self.value = tk.StringVar()

        self.combo = ttk.Combobox(
            self,
            textvariable=self.value,
            values=values,
            state="readonly",
            width=12
        )
        self.combo.pack(side="left")

        if initial is not None:
            self.value.set(initial)
        elif values:
            self.value.set(values[0])

        self.combo.bind("<<ComboboxSelected>>", self._selected)

    def _selected(self, event):
        if self.callback is not None:
            self.callback(self.value.get())

    def get(self):
        return self.value.get()

    def set(self, value):
        self.value.set(value)

if __name__ == "__main__":

    def frequency_changed(frequency):
        print(f"Frequency: {frequency:.3f} MHz")

    root = tk.Tk()
    root.title("Dropdown")


    def gain_changed(value):
        print("Gain:", value)


    gain_values = ["auto","0.0","10.0","20.0","30.0","40.0"]

    gain_control = DropDown(
        root,
        "Gain",
        gain_values,
        initial="20",
        callback=gain_changed
    )

    gain_control.pack(padx=20, pady=10)


    root.mainloop()