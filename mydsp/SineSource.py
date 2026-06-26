

class SineSource:

    description = "SineWave Source sample_rate = sample_rate, frequency = frequency, amplitude = amplitude, frame_size = frame_size,num_frames = num_frames"

    def __init__(self,sample_rate,frame_size,num_frames,frequency,amplitude):
        self.sample_rate=sample_rate
        self.frame_size=frame_size
        self.num_frames=num_frames
        self.frequency=frequency
        self.amplitude=amplitude


        

