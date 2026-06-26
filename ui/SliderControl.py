import tkinter as tk


class SliderControl:

    def __init__(
            self,
            parent,
            label,
            minval,
            maxval,
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
            resolution=0.1,
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