from mydsp import Generators
from mydsp.SigClasses import PlaySignals
from scipy.fftpack import fft, ifft
from mydsp.Filters import *

import matplotlib.pyplot as plt
from math import *
import time

frame_size = 1000
def hann(n):
    return sin(pi*n/frame_size)**2
w = np.zeros(frame_size)
for i in range(0,frame_size):
    w[i] = hann(i)

#scq = Signal.from_file("count.wav")

fs = 8000

#s1 = scq[0].extract(0,8000)
#s1 = Generators.sineWave("sine",0.01,200,fs,0,1.0)
s2 = Generators.sineWave("sine2",0.5,800,fs,0,1.0)
#s2 = scq[0]
#s2 = Generators.unitWave("unit",8000,0,1.0)
#print(s2.name)
#sin = s1.add(s2)
simpl = Generators.deltaWave("impulse",fs,0.01)
#s2 = simpl
#sin.plotFreq()
#sin.name="Input"
#sin.plotTime(tmin=0.12,tmax=0.13)
#sin.plotFreq()
f = SincLPF("sincf",8000,800.0,20)
#f = AnalogFilter("avg",np.array([.33,.33,.33]),np.array([1]))
#f = UnitFilter("unit",8000,800,1)
#print(f.a)
#print(sin.y[0:10])
#print(f.size)
#f = ButterworthFilter("b",800.0,8000.0,4)

#imp.plotTime()
#plt.figure()
#f.plot(1)
#s3.plotFreq(logscale=True)
#s3.plotTime()
#scq[0].plotTime()
#s5 = f.processfft(sin,frame_size)
s2.plotTime()
f.setWindowType(WindowType.HANN)
#s5 = f.processfft(s2,frame_size)
start_time = time.time()
s5 = f.processConv(s2,frame_size)
#s5 = f.process(s2)
end_time = time.time()
elapsed_time = end_time - start_time

#s5 = f.process(s2)
#s5 = f.process(s2)
s5.name="Output"
s5.plotFreq()
s5.plotTime(stick=False,tmin=0,tmax=0.14)
s5.plotTime()
#sin.plotTime(stick=False,tmax=0.2)
sout=[s5,s5]
#Signal.to_file(sout,"iir_out.wav")
print(f"Filter time: {elapsed_time:.6f} seconds")
f.plotWindow(1000)
plt.show()

