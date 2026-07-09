import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft

from mydsp.SigClasses import Signal
from mydsp.Utils import create_ola_function


class FFTFilter:
    name = None
    fs = 0
    filt = np.zeros(1)
    prevFrame = np.zeros(1)
    prevResult = np.zeros(1)
    envelope = np.zeros(1)


    def reset(self):

        self.prevFrame = np.zeros(1)
        self.prevResult = np.zeros(1)


    def getEnvelope(self,m,n):

        if self.envelope.size >  1:
            return self.envelope

        hwin = create_ola_function(m, n)
        self.envelope = np.array([hwin(i) for i in range(m)])
        return self.envelope


    def plotEnvelope(self,m):
        w = self.getEnvelope(m+self.overlap,self.overlap)

        N = len(w)  # Length of the array
        plt.figure(figsize=(8, 4))  # Set figure size
        plt.plot(np.arange(N), w, marker='o', linestyle='-')  # Plot with markers
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Plot of Array x")
        plt.grid(True)



    def processFrame(self,frame):

        M = self.overlap
        yin = frame
        if M != 0:
            yin = np.concatenate((self.prevFrame[-M:], frame))

        y = self.fft_convolution(yin)

        self.prevFrame = frame

        return y

    def doFrame(self,frame):

        if frame is None :
            return self.prevResult
        M = self.overlap
        prev_result_size = self.prevResult.size
        if self.prevFrame.size == 1:  # first time in previous results set to zero
            self.prevFrame = np.zeros(self.frame_size)
            self.prevResult = np.zeros(self.frame_size)
        env = self.getEnvelope(self.frame_size + M, M)
        yin = frame
        if M != 0:
            yin = np.concatenate((self.prevFrame[-M:], frame))

        y = self.fft_convolution(yin)*env

        yout = np.copy(self.prevResult)
        yout[-M:]+=y[:M]
        self.prevResult = y[M:]
        self.prevFrame = frame
# This is the first frame
        if prev_result_size == 1:
            return None
        return yout

    def fft_convolution(self,yin):

        zo = fft(yin)
        z1 = zo * self.filt
        yo = ifft(z1).real
        return yo

    def unitFrame(self, frame, hwin):

        N = frame.size  # Output size matches x
        y = np.zeros(N, dtype=np.float64)  # Initialize output array
        h = np.pad(self.a, (0, N - self.a.size), mode='constant')
        for n in range(N):
             y[n] += hwin[n] * frame[n]

        return y


    def processConv(self,signal_in):


        if self.prevFrame.size == 1:
            self.prevFrame = np.zeros(self.frame_size)
        M = self.overlap

        env = self.getEnvelope(self.frame_size + M, M)
        if self.a.size > 1 :
            self.filt = fft(self.a, self.frame_size+M)
        signal_out = Signal(self.name + "(" + signal_in.name + ")", signal_in.size, signal_in.dt)
        num_frames = int(signal_in.size/self.frame_size)

        for i in range(0,num_frames):

            frame = signal_in.y[i*self.frame_size:(i+1)*self.frame_size]
            yout = self.processFrame(frame)*env
            signal_out.add_with_offset(yout,i*self.frame_size-M)

        signal_out.x = signal_out.dt*np.array(range(0,signal_out.size))
        return signal_out





    def plotFFT(self,fa = 1.0,stick=False):


        N = self.frame_size + self.overlap
        Ny = int(N/2)


        # Compute the FFT of the impulse response
        #fft_values = np.fft.fft(impulse_response)
        fft_values = self.filt
        # Frequency axis (including negative frequencies)
        freq = np.fft.fftfreq(N, d=1 / self.fs)
        m = int(float(Ny) * fa)
        # Convert FFT values to magnitude and then to dB scale
        magnitude = np.abs(fft_values)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)  # Added small constant to avoid log(0)
        xvals = np.ones(N)
        xvals[:Ny] = freq[Ny:]
        xvals[Ny:] = freq[:Ny]
        yvals = np.ones(N)
        yvals[:Ny] = magnitude_db[Ny:]
        yvals[Ny:] = magnitude_db[:Ny]
        ymax = np.amax(yvals[Ny: Ny + m])
        ymin = np.amin(yvals[Ny:Ny + m])
        # Plot the frequency response
        plt.figure(figsize=(10, 6))
        plt.plot(xvals, yvals)
        plt.title('Frequency Response of the Impulse Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.xlim([-self.fs*fa / 2, self.fs*fa / 2])  # Limiting x-axis to show negative and positive frequencies
        plt.ylim(ymin, ymax)


    def impulse(self):
        n = self.a.size
        s = Signal("impulse-" + self.name, n , 1 / self.fs)
        for i in range(n ):
            s.x[i] = i / self.fs
            s.y[i] = self.a[i]

        return s