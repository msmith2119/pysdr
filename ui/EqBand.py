

import tkinter as tk


class EqBand:

    def __init__(
            self,
            parent,
            uid,
            label,
            minimum,
            maximum,
            initial,
            callback):

        self.callback = callback
        self.uid = uid
        self.frame = tk.Frame(parent)

        self.value_var = tk.StringVar(
            value=f"{initial:.1f}"
        )

        #
        # Current gain
        #
        tk.Label(
            self.frame,
            textvariable=self.value_var,
            width=6
        ).pack()

        #
        # Vertical slider
        #
        self.slider = tk.Scale(
            self.frame,
            from_=maximum,          # reverse so max is at top
            to=minimum,
            orient=tk.VERTICAL,
            showvalue=False,
            length=250,
            width=30,
            sliderlength=40,
            resolution=0.1,
            command=self._on_drag
        )

        self.slider.set(initial)
        self.slider.pack(pady=5)

        self.slider.bind(
            "<ButtonRelease-1>",
            self._on_release
        )

        #
        # Band label
        #
        tk.Label(
            self.frame,
            text=label
        ).pack()

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def _on_drag(self, value):
        self.value_var.set(f"{float(value):.1f}")

    def _on_release(self, event):
        self.callback(self.slider.get())