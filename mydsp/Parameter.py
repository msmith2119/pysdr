
from enum import Enum

class ParameterType(Enum):
    FLOAT = 1,
    INT = 2,
    BOOL = 3

class Parameter:
    def __init__(self,type,name,min,max,val):
        self.type = type
        self.name = name
        self.min = min
        self.max = max
        self.val = val




