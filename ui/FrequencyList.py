import tkinter as tk


class FrequencyList(tk.Frame):
    """
    Horizontal list of editable frequency fields.

    Each frequency is displayed in its own Entry widget.
    """

    def __init__(self, parent, frequencies=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.entries = []

        if frequencies:
            for frequency in frequencies:
                self.add(frequency)

    def add(self, frequency):
        """Add a frequency to the right end of the list."""
        entry = tk.Entry(self, width=12)
        entry.insert(0, str(frequency))

        entry.pack(side=tk.LEFT, padx=2)

        self.entries.append(entry)

    def pop(self):
        """Remove the last frequency from the list."""
        if self.entries:
            entry = self.entries.pop()
            entry.destroy()

    def remove(self, index):
        """Remove the frequency at index."""
        if 0 <= index < len(self.entries):
            entry = self.entries.pop(index)
            entry.destroy()

    def get(self):
        """Return the current contents as a list of strings."""
        return [entry.get() for entry in self.entries]

    def clear(self):
        """Remove all frequencies."""
        for entry in self.entries:
            entry.destroy()

        self.entries.clear()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Frequency List Test")

    frequency_list = FrequencyList(root)
    frequency_list.pack(padx=10, pady=10)

    frequency_list.add("104.500")
    frequency_list.add("107.700")
    frequency_list.add("121.900")

    tk.Button(
        root,
        text="Print List",
        command=lambda: print(frequency_list.get())
    ).pack(pady=5)

    root.mainloop()