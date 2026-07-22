

import tkinter as tk


class EqBand:

    def __init__(
            self,
            parent,
            uid,
            label,
            maximum,
            minimum,
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
            from_=minimum,          # reverse so max is at top
            to=maximum,
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

        self.slider.bind("<Button-4>", self._on_mousewheel_up)
        self.slider.bind("<Button-5>", self._on_mousewheel_dn)
        self.slider.bind("<Enter>", self._enter)
        #
        # Band label
        #
        tk.Label(
            self.frame,
            text=label
        ).pack()

    def setValue(self,value):
        self.slider.set(value)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def _on_drag(self, value):
        self.value_var.set(f"{float(value):.1f}")

    def _on_release(self, event):
        self.callback(self.slider.get())

    def _enter(self, event):
        self.active = True

    def _leave(self, event):
        self.active = False

    def _on_mousewheel_up(self, event):
        if not self.active:
            return

        value = self.slider.get()


        value += self.slider.cget("resolution")


        # clamp to range
        value = min(
            self.slider.cget("from"),
            max(self.slider.cget("to"), value)
        )

        self.slider.set(value)
        self.value_var.set(f"{value:.1f}")
        self.callback(value)

    def _on_mousewheel_dn(self, event):
        if not self.active:
            return


        value = self.slider.get()


        value -= self.slider.cget("resolution")

        # clamp to range
        value = max(
            self.slider.cget("to"),
            min(self.slider.cget("from"), value)
        )

        self.slider.set(value)
        self.value_var.set(f"{value:.1f}")
        self.callback(value)