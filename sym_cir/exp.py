
from mydsp.OscillatorSource import OscillatorSource
from mydsp.Utils import plotFFT, plot_array
from matplotlib import pyplot as plt
from mydsp.FreqShiftFilter import FreqShiftFilter
from scipy.fftpack import fft, ifft
import numpy as np

fs = 250000.0
frame_size=10000

osc = OscillatorSource("test",fs,frame_size,38000,1.0,0,0)

fsh = FreqShiftFilter("ff",fs,frame_size,37000,15000.0)
block = osc.getFrame()
frame = block[:,0]
fosc = fft(frame)
fout  =  fsh.doFrame(frame)
fosc_mag = np.abs(fosc)
#fout_mag = np.abs(fout)

plotFFT(fout,fs)
plt.show()

