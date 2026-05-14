set frame_size 1000
set sample_rate 8000
source wav mywav [path=audio/cqi2.wav,frame_size=frame_size]
filter Notch mynotch [fs=8000,fc=1000,fbw=100,frame_size=1000]
filter LP mylp [fs=8000,fc=800,frame_size=1000]
#filter Unit unitF [frame_size=1000]
sink WavFile myout [path=audio/cqi2_out.wav,sample_rate=sample_rate,frame_size=frame_size]
connect mypipe mywav (mynotch,mylp) to myout
