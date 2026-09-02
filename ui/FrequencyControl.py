

import tkinter as tk


class FrequencyControl(tk.Frame):
    """
    Three digits before and three digits after the decimal point.

    Example:
        101.705 MHz

    Keyboard behavior:
        0-9     Enter digit and advance
        Left    Move to previous digit
        Right   Move to next digit
        Home    Move to first digit
        End     Move to last digit
        Delete  Clear current digit
        Backspace clear current digit, or move left if already clear
    """

    def __init__(self, parent, frequency=100.000,**kwargs):
        super().__init__(parent,**kwargs)


        self.entries = []

        # Six editable digits: xxx.xxx
        digits = self._frequency_to_digits(frequency)

        for i in range(3):
            entry = self._create_digit_entry(digits[i])
            entry.grid(row=0, column=i, padx=1)
            self.entries.append(entry)

        # Decimal point
        decimal = tk.Label(
            self,
            text=".",
            font=("TkDefaultFont", 14)
        )
        decimal.grid(row=0, column=3)

        for i in range(3, 6):
            entry = self._create_digit_entry(digits[i])
            entry.grid(row=0, column=i + 1, padx=1)
            self.entries.append(entry)


        self.setFrequency(frequency)

    def _create_digit_entry(self, digit):

        entry = tk.Entry(
            self,
            width=2,
            justify="center",
            fg="black",
            bg="white"
        )
        entry.insert(0, digit)

        # Keyboard handlers
        entry.bind("<KeyPress>", self._key_press)
        entry.bind("<Left>", self._move_left)
        entry.bind("<Right>", self._move_right)
        entry.bind("<Home>", self._move_home)
        entry.bind("<End>", self._move_end)
        entry.bind("<BackSpace>", self._backspace)
        entry.bind("<Delete>", self._delete)
        entry.bind("<FocusIn>", self._focus_in)
        entry.bind("<FocusOut>", self._focus_out)
        # Clicking an entry should select its digit.
        entry.bind("<Button-1>", self._mouse_click)

        return entry

    # ------------------------------------------------------------------
    # Keyboard handling
    # ------------------------------------------------------------------

    def _key_press(self, event):
        if event.char and event.char.isdigit():
            entry = event.widget

            # Replace whatever is currently there.
            entry.delete(0, tk.END)
            entry.insert(0, event.char)

          #  self._frequency_changed()

            # Move to next digit.
            self._move_to_entry(entry, +1)

            return "break"

        # Don't allow any other character into the Entry.
        return "break"

    def _focus_in(self, event):
        event.widget.configure(
            fg="blue",
            bg="light yellow"
        )

    def _focus_out(self, event):
        event.widget.configure(
            fg="black",
            bg="white"
        )
    def _move_left(self, event):
        self._move_to_entry(event.widget, -1)
        return "break"

    def _move_right(self, event):
        self._move_to_entry(event.widget, +1)
        return "break"

    def _move_home(self, event):
        self.entries[0].focus_set()
        self.entries[0].selection_range(0, 1)
        return "break"

    def _move_end(self, event):
        self.entries[-1].focus_set()
        self.entries[-1].selection_range(0, 1)
        return "break"

    def _backspace(self, event):
        entry = event.widget

        if entry.get():
            entry.delete(0, tk.END)
        else:
            self._move_to_entry(entry, -1)

        self._frequency_changed()
        return "break"

    def _delete(self, event):
        event.widget.delete(0, tk.END)
        self._frequency_changed()
        return "break"

    def _mouse_click(self, event):
        # Let Tk establish focus, then select the digit.
        entry = event.widget
        self.after_idle(lambda: entry.selection_range(0, 1))

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _move_to_entry(self, entry, direction):
        try:
            index = self.entries.index(entry)
        except ValueError:
            return

        index += direction

        if 0 <= index < len(self.entries):
            new_entry = self.entries[index]
            new_entry.focus_set()
            new_entry.selection_range(0, 1)

    # ------------------------------------------------------------------
    # Frequency conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _frequency_to_digits(frequency):
        """
        Convert MHz frequency to six display digits.

        101.705 -> ['1', '0', '1', '7', '0', '5']
        """

        # Work in kHz to avoid floating-point display problems.
        khz = int(round(frequency * 1000))

        if khz < 0 or khz > 999999:
            raise ValueError("Frequency must be between 0.000 and 999.999 MHz")

        return f"{khz:06d}"

    def getFrequency(self):
        """
        Return the current frequency in MHz.
        """

        digits = ''.join(
            entry.get() if entry.get() else '0'
            for entry in self.entries
        )

        khz = int(digits)
        return khz / 1000.0

    def setFrequency(self, frequency):
        """
        Set the displayed frequency in MHz.
        """

        digits = self._frequency_to_digits(frequency)

        for entry, digit in zip(self.entries, digits):
            entry.delete(0, tk.END)
            entry.insert(0, digit)

    def _frequency_changed(self):
        if self.callback is not None:
            self.callback(self.getFrequency())


# ----------------------------------------------------------------------
# Simple test program
# ----------------------------------------------------------------------

if __name__ == "__main__":

    def frequency_changed(frequency):
        print(f"Frequency: {frequency:.3f} MHz")

    root = tk.Tk()
    root.title("Frequency Control")

    control = FrequencyControl(
        root,
        frequency=101.705,
    )

    control.pack(padx=20, pady=20)

    root.mainloop()

