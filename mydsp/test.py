from mydsp.NotchFilter import NotchFilter
from mydsp.SigClasses import description

#f = NotchFilter("aa",8000,1000,101,1000)

class_name = "NotchFilter"
print(globals().keys())
cl = globals().get(class_name)
description = getattr(cl,"description")
print(type(cl))
print(description)