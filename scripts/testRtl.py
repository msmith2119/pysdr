

from mydsp.RtlSdrSource import RtlSdrSource
from rtlsdr import RtlSdr
import time
fs = 240000
frame_size = 8192
freq=90.3e6
rtl =  RtlSdrSource("myrtl",fs,freq,frame_size,10)
rtl.start()
for i in range(0,10):
    start = start = time.perf_counter()
    frame = rtl.getFrame()
    elapsed = time.perf_counter() - start
    print("elapsed time:", elapsed)
    time.sleep(0.03)


