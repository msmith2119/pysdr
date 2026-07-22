import tkinter as tk


class SliderControl:

    def __init__(
            self,
            parent,
            label,
            minval,
            maxval,
            resolution,
            initial,
            callback):

        self.callback = callback

        frame = tk.Frame(parent)
        frame.pack(fill="x", padx=10, pady=15)

        tk.Label(
            frame,
            text=label,
            width=12,
            anchor="w"
        ).grid(row=0, column=0, sticky="w")

        self.value_var = tk.StringVar(
            value=f"{initial:.1f}"
        )

        self.slider = tk.Scale(
            frame,
            from_=minval,
            to=maxval,
            orient=tk.HORIZONTAL,
            showvalue=False,
            sliderlength=50,
            resolution=resolution,
            width=30,
            length=300,
            command=self._on_drag
        )


        self.slider.set(initial)

        self.slider.grid(
            row=0,
            column=1,
            sticky="ew"
        )

        self.slider.bind(
            "<ButtonRelease-1>",
            self._on_release
        )
        self.slider.bind("<Button-4>", self._on_mousewheel_up)
        self.slider.bind("<Button-5>", self._on_mousewheel_dn)
        self.slider.bind("<Enter>", self._enter)
        tk.Label(
            frame,
            textvariable=self.value_var,
            width=8,
            anchor="e"
        ).grid(
            row=0,
            column=2,
            padx=(10, 0)
        )

        frame.columnconfigure(1, weight=1)



    def _on_drag(self, value):
        """Update displayed value while dragging."""
        self.value_var.set(f"{float(value):.1f}")

    def _on_release(self, event):
        """Notify caller when user releases mouse."""
        value = self.slider.get()
        self.callback(value)

    def _enter(self, event):
        self.active = True

    def _leave(self, event):
        self.active = False

    def _on_mousewheel_up(self, event):
        if not self.active:
            return


        value = float(self.slider.get())

        delta = self.slider.cget("resolution")*2.0

        value +=delta

        # clamp to range
        value = min(
            self.slider.cget("to"),
            max(self.slider.cget("from"), value)
        )



        self.slider.set(value)

        self.value_var.set(f"{value:.1f}")
        self.callback(value)

    def _on_mousewheel_dn(self, event):
        if not self.active:
            return


        value = self.slider.get()
        delta = self.slider.cget("resolution")*2.0

        value -= delta

        # clamp to range
        value = min(
            self.slider.cget("to"),
            max(self.slider.cget("from"), value)
        )

        self.slider.set(value)
        self.value_var.set(f"{value:.1f}")
        self.callback(value)