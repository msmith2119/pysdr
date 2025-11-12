import time

from mydsp import Generators, Filters, SigClasses
from mydsp.FFTLPFilter import FFTLPFilter
from mydsp.FFTNotchFilter import FFTNotchFilter
from mydsp.Filters import *
from mydsp.LPFilter import LPFilter
from mydsp.NotchFilter import NotchFilter
from mydsp.SincLPFilter import SincLPFilter
from mydsp.WavFileSink import WavFileSink
from mydsp.WavFileSource import WavFileSource

frame_size = 1000
fs = 8000

scq = Signal.from_file("audio/cqi.wav")
#s2 = Generators.sineWave("sine2",0.5,1200,fs,0,1.0)
s2 = scq[0]

#f = SincLPFilter("sincf", 8000, 800.0, 20,1000)
#f = LPFilter("fft",8000,800,frame_size)
#f = FFTNotchFilter("notch",fs,1000,100,frame_size + 200)
#f = AnalogFilter("simp",fs,np.array([1,1]),np.array([1]))
f = NotchFilter("notch",fs,1000,100,frame_size)
#f.impulse().plotTime()
#f.plot(1)
f.plotFFT()
#start_time = time.time()
#s5 = f.processConv(s2,frame_size)
s2.frame_size = frame_size


source = WavFileSource("audio/cqi.wav",frame_size)
sink = WavFileSink("iir_out.wav", frame_size, sample_rate=fs)
while True:
    frame = source.getFrame()
    if frame is None:
          break
    out_frame = f.doFrame(frame)
    if out_frame is not None:
         sink.writeFrame(out_frame)
sink.writeFrame(f.prevResult)
sink.close()
source.close()

#s5 = f.process(s2)
#s5 = f.processConv(s2)
#end_time = time.time()
#elapsed_time = end_time - start_time

#s5.name="Output"
#s5.plotFreq()

#s5.plotTime(stick=False,tmin=0,tmax=0.14)
#s5.plotTime()
#sout=[s5,s5]
#Signal.to_file(sout,"iir_out.wav")
#print(f"Filter time: {elapsed_time:.6f} seconds")
#f.plotEnvelope(1000)
plt.show()

