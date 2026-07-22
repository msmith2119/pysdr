
import re
from sympy import Float, Symbol
from sympy.core.expr import Expr


from sym_cir.core.Capacitor import Capacitor
from sym_cir.core.Circuit import Circuit
from sym_cir.core.Pot import Pot
from sym_cir.core.Resistor import Resistor


class SpiceParser:
    def __init__(self):

        self.circuit = Circuit()



    _suffixes = {
        "T": 1e12,
        "G": 1e9,
        "Meg": 1e6,
        "k": 1e3,
        "m": 1e-3,
        "u": 1e-6,
        "n": 1e-9,
        "p": 1e-12,
        "f": 1e-15,
    }

    _number_re = re.compile(
        r'^([+-]?'
        r'(?:\d+(?:\.\d*)?|\.\d+)'
        r'(?:[eE][+-]?\d+)?)'
        r'([A-Za-z]*)$'
    )




    def parseFile(self,filename):
        with open(filename) as f:
            for line in f:
                dev = self.parseLine(line)
                if dev is not None:
                    self.circuit.addDevice(dev)
        return self.circuit

    def parseLine(self,line):

        dev = None
        if line[0] == 'R' :
            dev = self.parseR(line)
        elif line[0] == 'P' :
            dev = self.parseP(line)
        elif line[0] == 'C' :
            dev = self.parseC(line)
        return dev
    @classmethod
    def parse_value(cls, value):
        """
        Convert a SPICE value to a SymPy object.

        Examples
        --------
        "10k"      -> Float(10000.0)
        "2.2u"     -> Float(2.2e-6)
        "100"      -> Float(100.0)
        "RLOAD"    -> Symbol("RLOAD")
        "gain"     -> Symbol("gain")
        """

        if isinstance(value, Expr):
            return value

        if isinstance(value, (int, float)):
            return Float(value)

        text = str(value).strip()

        m = cls._number_re.match(text)

        if not m:
            return Symbol(text)

        number = float(m.group(1))
        suffix = m.group(2)

        if suffix == "":
            return Float(number)

        if suffix not in cls._suffixes:
            # Not a recognized SPICE suffix;
            # treat the whole thing as a symbolic identifier.
            return Symbol(text)

        return Float(number * cls._suffixes[suffix])


    def parseR(self,line):
        match = re.match(r"^(R\w+)\s+(\w+)\s+(\w+)\s+(\S+)",line)
        if match:
            name = match.group(1)
            nn1 = match.group(2)
            nn2 = match.group(3)
            vv = match.group(4)
            n1 = self.circuit.getNode(nn1)
            n2 = self.circuit.getNode(nn2)
            val = SpiceParser.parse_value(vv)
            r = Resistor(name,n1,n2,val)
            return r
        else:
            print(f"ERROR : SPICE line unparseable : {line}")
            return None

    def parseP(self,line):
        match = re.match(r"^(P\w+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\S+)\s+(\w+)",line)
        if match:
            name = match.group(1)
            nn1 = match.group(2)
            nn2 = match.group(3)
            nn3 = match.group(4)
            vv = match.group(5)
            param = match.group(6)

            n1 = self.circuit.getNode(nn1)
            n2 = self.circuit.getNode(nn2)
            n3 = self.circuit.getNode(nn3)
            val = SpiceParser.parse_value(vv)
            parm = SpiceParser.parse_value(param)
            r = Pot(name,n1,n2,n3,val,parm)
            return r
        else:
            print(f"ERROR : SPICE line unparseable : {line}")
            return None


    def parseC(self,line):
        match = re.match(r"^(C\w+)\s+(\w+)\s+(\w+)\s+(\S+)",line)
        if match:
            name = match.group(1)
            nn1 = match.group(2)
            nn2 = match.group(3)
            vv = match.group(4)
            n1 = self.circuit.getNode(nn1)
            n2 = self.circuit.getNode(nn2)
            val = SpiceParser.parse_value(vv)
            c = Capacitor(name, n1, n2, val)
            return c
        else:
            print(f"ERROR : SPICE line unparseable : {line}")
            return None
