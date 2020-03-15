import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np


wav_file_name = ('piano.wav')

samplerate, data = wavfile.read(wav_file_name)

data = data[:, 0] #dropping our signal data to one channel (the left side) LIST SLICING works like -->a[start:end]
#data = data.ravel() I don't think we need to ravel our data, dropping it to the left side already did this
print(f"data = {data}")
print(f"data shape = {data.shape[0]}") #will print the shape of our data, specifically, how many items in that specific channel (n rows up and down, m columns side to side)


length = data.shape[0] / samplerate #takes the first item in our data shape (in this case, 264600), and divides it by our sampling rate to get the total amount of time the clip runs for


#time = np.linspace(0., length, data.shape[0]) #start at 0, stop at 6,
#plt.plot(time, data, label="Left channel")

plt.specgram(data,Fs=samplerate) #plotting the spectrogram data https://matplotlib.org/3.2.0/api/_as_gen/matplotlib.pyplot.specgram.html
#Fs = the sampling frequency (the number of samples per unit time), when using samplerate, it will show the frequencies with respect to the time of the clip
#window = A function or vector of length NFFT. To create window vectors see window_hanning, window_none, numpy.blackman, numpy.hamming, numpy.bartlett, scipy.signal, scipy.signal.get_window. Default is window_hanning. If a function is passed as the argument, it must take a data segment as an argument and return the windowed version of the segment.
#pad_to = The number of points to which the data segment is padded when performing the FFT. Doesn't increase the actual resolution of the spectrum (the minimum distance between resolvable peaks), but can give more points in the plot, allowing for more detail. This corresponds to the n parameter in the call to fft(). The default is None, which sets pad_to equal to NFFT
#NFFT = number of data points used in each block for the FFT, default value is 256


plt.xlabel("Time [s]")
plt.ylabel("Frequency")
plt.show()
