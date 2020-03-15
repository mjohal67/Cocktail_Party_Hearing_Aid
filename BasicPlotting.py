import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np


wav_file_name = ('piano.wav')

samplerate, data = wavfile.read(wav_file_name)

print (f"type of data = {data.dtype}")
print(f"data = {data}")
print(f"number of channels = {data.shape[1]}")
print(f"data shape = {data.shape[0]}") #will print the shape of our data, (n rows up and down, m columns side to side)


print (f"sampling rate = {samplerate}")

length = data.shape[0] / samplerate #takes the first item in our data shape (in this case, 264600), and divides it by our sampling rate to get the total amount of time the clip runs for
print(f"length = {length}s")


import matplotlib.pyplot as plt
import numpy as np
time = np.linspace(0., length, data.shape[0])
plt.plot(time, data[:, 0], label="Left channel") #dropping signal to mono channel, takes our dual-channel signal and ISOLATES THE LEFT CHANNEL (the first element in each tuple)
#plt.specgram(data,Fs=samplerate)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()
