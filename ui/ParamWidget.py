
import tkinter as tk
from functools import partial

from mydsp.Parameter import ParameterType
from mydsp.Parameter import Parameter
from ui.SliderControl  import SliderControl


class ParamWidget(tk.Frame):
    def __init__(self, root,name, params,N, callback, **kwargs):

        super().__init__(root, **kwargs)

        self.name = name
        self.root = root
        self.params = params
        self.callback = callback


        for param in params:
            cval = param.val
            df = float(cval / N)
            print(f"df={df}")
            min = cval - (N / 2) * df
            print("min = {min}")
            max = cval + (N / 2) * df
            print("max = {max}")
            resolution = df

            def value_changed(pname, value):
                callback(pname,value)
            fmt = "0.0f"
            if cval < 1:
                fmt = "0.3f"

            SliderControl(
                self,
                param.name,
                min,
                max,
                resolution,
                param.val,
                partial(value_changed, param.name),
                format_spec = fmt
            )
        tk.Button(
            self,
            text="Close",
            command=root.destroy
        ).pack(pady=10)


if __name__ == "__main__":

    fname = "F1"
    def param_changed(name,value):
        print(f"{fname} param changed {name} = {value} ")

    def do_reset():
        print("do reset")
    root = tk.Tk()
    root.title("Param Control")
    params = []
    params.append(Parameter(ParameterType.FLOAT,"fc",0,5000,0.1))
    param_editor = ParamWidget(
      root,
    name="param_widget",
    params=params,
    N=100,
    callback=param_changed
    )

    param_editor.pack(padx=20, pady=20)
    tk.Button(
        root,
        text="Reset",
        command=do_reset
    ).pack(pady=10)
    root.mainloop()
