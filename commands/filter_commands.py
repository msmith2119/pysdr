import importlib

import mydsp
from mydsp.NotchFilter import NotchFilter
from .dsl_globals import get_context

class FilterCommands:

    def cmd_filter(self, args):
        if len(args) < 2:
            print("Usage: filter <type> <name> [params]")
            return
        filter_type, filter_name = args[0], args[1]
        param_str = " ".join(args[2:])

        try:
            params = {}
            params['name']=filter_name
            if param_str:
                for item in param_str.split():
                    k, v = item.split('=')
                    params[k] = eval(v,get_context().vars )


            filter_class = globals().get(filter_type.capitalize() + "Filter")
            if not filter_class:
                print(f"Unknown filter type: {filter_type}")
                return

            f = filter_class(**params)
            self.filters[filter_name] = f
            print(f"Filter '{filter_name}' created.")

        except Exception as e:
            print(f"Error creating filter: {e}")

    def cmd_filters(self, args):
        if not self.filters:
            print("No Filters defined.")
            return
        for name in self.filters:
            print(f"- {name}")

    def cmd_filtertype(self, args):
        if len(args) != 1:
            print("Usage: filtertype <type>")
            return

        class_name = args[0].capitalize() + "Filter"
        filter_class =   globals().get(class_name)
        print(type(filter_class))
        print(filter_class)
        if not filter_class:
            print(f"Filter class '{class_name}' not found.")
            return
        desc = getattr(filter_class, "description", None)
        if desc:
            print(f"{class_name}: {desc}")
        else:
            print(f"{class_name} exists but has no description.")
