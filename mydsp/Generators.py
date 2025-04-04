import math
import random
import numpy as np
from .SigClasses import Signal

def deltaFunc():
    def func(x):
        if x == 0:
            return 1.0
        else:
            return 0.0
    return func
def unitFunc():
    def func(x):
        return 1.0
    return func
def sinFunc(a,f,phi=0):
    def func(x):
        return a*math.sin(2*math.pi*f*x + phi)
    return func

def cosFunc(a,f,phi=0):
    def func(x):
        return a*math.cos(2*math.pi*f*x + phi)
    return func

def randFunc(yl,yh):
    def func(x):
        return yl + (yh-yl)*random.random()
    return func

def squareFunc(a,T):
     def func(x):
         xm = int(x/T)*T
         print("xm",xm)
         xp = x - xm
         print(xp)
         if xp < T/2:
             val = a
         else:
             val = 0
         return val
     return func


def genNoise(name,fs,n):
    dt = 1/fs

    s = Signal(name,n,dt)


    s.x = dt*np.arange(n)
    s.y =  np.clip(np.random.normal(loc=0, scale=0.01, size=n),a_min=-1.0,a_max=1.0)
    return s


def genWave(name,func,fs,xl,xh):
    dt = 1 / fs
    n = int((xh - xl) / dt)
    s = Signal(name, n, dt)
    for i in range(n):
        s.x[i] = xl + dt * i
        s.y[i] = func(s.x[i])
    return s

def unitWave(name,fs,xl,xh):
    return genWave(name,unitFunc(),fs,xl,xh)
def sineWave(name,ampl,f,fs,xl,xh,phi=0):
    return genWave(name,sinFunc(ampl,f,phi),fs,xl,xh)

def cosWave(name,ampl,f,fs,xl,xh,phi=0):
    return genWave(name,cosFunc(ampl,f,phi),fs,xl,xh)

def randWave(name,yl,yh,fs,xl,xh):
    return genWave(name,randFunc(yl,yh),fs,xl,xh)

def squareWave(name,a,T,xh,fs):
    return genWave(name,squareFunc(a,T),fs,0,xh)

def deltaWave(name,fs,xh):
    return genWave(name,deltaFunc(),fs,0,xh)