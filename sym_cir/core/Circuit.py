


from sympy import *

from sym_cir.core.Node import Node


class Circuit:
    def __init__(self):

        self.devices = []
        self.nodes = {}
        self.counter = 0
        gnd_node = Node("gnd",-1)
        self.nodes[gnd_node.name] = gnd_node
        gnd_node = Node("0",-1)
        self.nodes[gnd_node.name] = gnd_node


    def getNode(self,name):
        if name in self.nodes:
            return self.nodes[name]

        else:
            n = Node(name,self.counter)
            self.nodes[name] = n
            self.counter +=1
            return n

    def getRank(self):
        return len(self.nodes)-2

    def addDevice(self, device):

        self.devices.append(device)

    def stamp_devices(self):

        n = self.getRank()
        self.Y = Matrix.zeros(n,n)

        for device in self.devices:
           device.stamp(self.Y)

    def getTwoPortY(self,inp,out):

        n = self.getRank()
        node = self.nodes.get(inp,None)
        if node == None:
            print(f"ERROR invalid node name : {inp}")
            return None
        i = node.value
        node = self.nodes.get(out,None)
        if node == None:
            print(f"ERROR invalid node out : {out}")
        j = node.value
        ports = [i, j]
        internal = [k for k in range(n) if k not in ports]


        Ypp = self.Y.extract(ports, ports)
        Ypr = self.Y.extract(ports, internal)
        Yrp = self.Y.extract(internal, ports)
        Yrr = self.Y.extract(internal, internal)

        Y2 = Ypp - Ypr * Yrr.inv() * Yrp
        return Y2

    def getTwoPortT(self,inp,out):

        y = self.getTwoPortY(inp,out)
        a = -y[1,1]/y[1,0]
        b = -1/y[1,0]
        c = -y.det()/y[1,0]
        d = -y[0,0]/y[1,0]

        T = Matrix([[a,c],[c,d]])
        return T