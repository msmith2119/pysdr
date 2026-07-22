


pattern = r"^(P\w+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\S+)\s+(\w+)"

class Pot:

    def __init__(self, name, n1, n2,n3, resistance,variable):
        self.name = name
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.R = resistance
        self.var =variable

    @property
    def nodes(self):
        return (self.n1, self.n2,self.n3)

    def stamp(self,ymatrix):
        i1 = self.n1.value
        i2 = self.n2.value
        i3 = self.n3.value
        g1 = 1/(self.R*self.var)
        g2 = 1/(self.R*(1-self.var))
        g = 1/(self.R*self.var)
        if i1  > -1 and i2  > -1:
            ymatrix[i1,i2] -= g1
            ymatrix[i2,i1] -= g1
        if i2 > -1 and i3 > -1:
            ymatrix[i2,i3] -= g2
            ymatrix[i3,i2] -= g2

        if i1 > -1:
            ymatrix[i1,i1] += g1
        if i2  > -1:
            ymatrix[i2,i2] += (g1+g2)
        if i3 > -1:
            ymatrix[i3,i3] += g2

    def toStr(self):
        return f"{self.name} {self.n1.value} {self.n2.value} {self.n3.value} {self.R} {self.var}"
