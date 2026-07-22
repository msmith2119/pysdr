
from sympy import *

from mydsp.Utils import store_numpy_array
from sym_cir.core.SpiceParser import SpiceParser
import numpy as np

parser = SpiceParser()



circuit = parser.parseFile("cir/simptone.cir")
circuit.stamp_devices()
y = circuit.getTwoPortY("in","out")

#print(N(circuit.Y,5))
T = circuit.getTwoPortT("in","out")
#print(N(T,5))
a  = 1/T[0,0]
r,c,T,s,z = symbols('r c T s z')
#print(cancel(together(a)))
N,D = fraction(cancel(together(a)))
print(N)
print(D)
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
print(a[0])
subs = {
    r: 1000,
    c: 1e-6,
    T: 1/8000
}
aa = [expr.subs(subs) for expr in a]
#coeffs_a = np.array([float(x) for x in aa])
bb = [expr.subs(subs) for expr in b]
#coeffs_b = np.array([float(x) for x in bb])


print(aa)
print(bb)

#store_numpy_array(coeffs_a,"../filters/rc.a")
#store_numpy_array(coeffs_b,"../filters/rc.b")

#c1 = 1e-08
#c2 = 2e-8
print(bb[4])
a = np.zeros(5)
b = np.zeros(5)
r1 = 50000.0
r2 = 50000.0

a[0] = 4.0e-8*r1*r2 + 0.000375*r1 + 0.00025*r2 + 0.78125
a[1] =0.00075*r1 + 0.0005*r2 + 3.125
a[2] =-8.0e-8*r1*r2 + 4.6875
a[3] =-0.00075*r1 - 0.0005*r2 + 3.125
a[4]= 4.0e-8*r1*r2 - 0.000375*r1 - 0.00025*r2 + 0.78125

b[0] =4.0e-8*r1*r2 + 0.000125*r1 + 0.00025*r2 + 0.78125
b[1] =0.00025*r1 + 0.0005*r2 + 3.125
b[2] =-8.0e-8*r1*r2 + 4.6875
b[3] =-0.00025*r1 - 0.0005*r2 + 3.125
b[4] =4.0e-8*r1*r2 - 0.000125*r1 - 0.00025*r2 + 0.78125

store_numpy_array(a,"../filters/simptone.a")
store_numpy_array(b,"../filters/simptone.b")
