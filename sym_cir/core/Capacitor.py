
from sympy import *

class Capacitor:

    def __init__(self, name, n1, n2, capacitance):
        self.name = name
        self.n1 = n1
        self.n2 = n2
        self.C = capacitance

    @property
    def nodes(self):
        return (self.n1, self.n2)


    def stamp(self,ymatrix):

        i1 = self.n1.value
        i2 = self.n2.value

        g = Symbol("s")*self.C

        if i1 > -1 and i2 > -1:
            ymatrix[i1, i2] -= g
            ymatrix[i2, i1] -= g
        if i1 > -1:
            ymatrix[i1, i1] += g
        if i2 > -1:
            ymatrix[i2, i2] += g

    def toStr(self):
        return f"{self.name} {self.n1.value} {self.n2.value} {self.C}"