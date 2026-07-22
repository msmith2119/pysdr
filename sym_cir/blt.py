
from sympy import *
from SparseArray import  SparseArray
import math
import numpy as np

R,C,T,s,z = symbols('R C T s z')

H = 1/(1+s*R*C)

Hz = H.subs(
    s,
    2*(z-1)/(T*(z+1))
)

Hz = cancel(together(Hz))

num, den = Hz.as_numer_denom()

order = Poly(1+s*R*C, s).degree()

num *= (z+1)**order
den *= (z+1)**order

num = expand(num)
den = expand(den)
#print(num)
#print(den)
a_p = Poly(den,z).all_coeffs()
b_p = Poly(num,z).all_coeffs()
print(a_p)
print(b_p)

c = 1e-07

fo =  500.0
r = 1/(2*math.pi*c*fo)
fs = 8000.0
subs = {
    R: r,
    C: c,
    T: 1/fs
}

print(r)
a = [expr.subs(subs) for expr in a_p]
b = [expr.subs(subs) for expr in b_p]

afile = "filters/rc.a"
bfile = "filters/rc.b"

with open(afile, "wb") as f:
    np.save(f, a)
with open(bfile, "wb") as f:
    np.save(f, b)

with open(afile, "rb") as f:
    a1 = np.load(f)
with open(bfile, "rb") as f:
    b1 = np.load(f)
print(a1)
print(b1)

