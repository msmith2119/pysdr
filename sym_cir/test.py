
from sympy import *
from zmq import devices
from scipy import signal
from mydsp.Utils import store_numpy_array
from sym_cir.core.SpiceParser import SpiceParser
import numpy as np

parser = SpiceParser()

circuit = parser.parseFile("cir/tone.cir")

circuit.stamp_devices()
#print(circuit.Y)q
#Y = circuit.Y
s,a1,a2 = symbols('s a1 a2')
print(circuit.nodes['in'].value)
print(circuit.nodes['out'].value)
#for i in range(Y.rows):
#    for j in range(Y.cols):
#        if Y[i,j] != 0 :
#            print(f"Y[{i},{j}] = {Y[i,j]}")

Y = circuit.Y

print(Y.det())
#Z = Y.inv()
#zz = Z.col(0)
#print(zz[0])
#print(zz[4])
#a = zz[4]/zz[0]
#print(a)
#N,D = fraction(cancel(together(a)))
#print(N)
#print(D)


#print(D)
input("Press Enter to continue...")

y = circuit.getTwoPortY("in","out")
fs = 8000.0

#print(N(circuit.Y,5))
T = circuit.getTwoPortT("in","out")
#print(N(T,5))
a  = 1/T[0,0]
print(a)
input("Press Enter to continue...")
r,c,T,s,z = symbols('r c T s z')
#print(cancel(together(a)))
N,D = fraction(cancel(together(a)))
print(N)
print(D)
a = Poly(D,s).all_coeffs()
b = Poly(N,s).all_coeffs()
print(a)
print(b)
print("*******************8")
bb_z, aa_z = signal.bilinear(b, a, fs)
print(bb_z)
print(aa_z)
store_numpy_array(aa_z,"../filters/tone.a")
store_numpy_array(bb_z,"../filters/tone.b")
input("Press Enter to continue...")
H = N/D
print(H)
input("Press Enter to continue...")
Hz = H.subs(
    s,
    2*(z-1)/(T*(z+1))
)

Hz = cancel(together(Hz))

num, den = Hz.as_numer_denom()


order  = Poly(D,s).degree()

num *= (z+1)**order
den *= (z+1)**order

num = expand(num)
den = expand(den)
#print(num)
#print(den)
print(Poly(num,z))
print(Poly(den,z))

a = Poly(den,z).all_coeffs()
b = Poly(num,z).all_coeffs()
a1 = [simplify(expr) for expr in a]
b1 = [simplify(expr) for expr in b]
print(a1)
print(b1)

input("Press Enter to continue...")

subs = {
    r: 1000,
    c: 1e-6,
    T: 1/8000
}
aa = [expr.subs(subs) for expr in a]
#coeffs_a = np.array([float(x) for x in aa])
bb = [expr.subs(subs) for expr in b]
#coeffs_b = np.array([float(x) for x in bb])

aa = [float(e)*1e+37 for e in aa]
bb = [float(e)*1e+37 for e in bb]
print(aa)
print(bb)
store_numpy_array(aa,"../filters/tone.a")
store_numpy_array(bb,"../filters/tone.b")

