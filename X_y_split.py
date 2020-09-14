import numpy as np
import librosa
from librosa import display
import os
from scipy.io.wavfile import write


completed = 0

X = []
y = []

def split_training_data():
    global X
    global y
    global completed
    global training_data
    global standardized_mix_stft
    global stft_mix_magnitude
    global features

    training_data = np.load("mix_irm_training_data_complex_good.npy")
    print ("Training Data Loaded")

    for features, label in training_data:
        #mix_angle = np.angle(stft_mix)
        #mix_phase = np.exp(1.0j* mix_angle)

        stft_mix_magnitude = np.array(abs(features))

        stft_mix_mean = np.mean(stft_mix_magnitude)
        stft_mix_stndrd = np.std(stft_mix_magnitude)
        standardized_mix_stft = (stft_mix_magnitude - stft_mix_mean) / stft_mix_stndrd

        X.append(standardized_mix_stft)
        y.append(label)
        completed += 1
        print (int(completed), "SAMPLES PROCESSED")

split_training_data()

'''
#TESTING--------
inverse_mix = librosa.core.istft(standardized_mix_stft, hop_length=int(512//2))
write("inverse_mix_X_TEST.wav", 16000, (inverse_mix).astype(np.float32))
#---------------
'''

X = np.array(X).reshape(-1, len(training_data[0][0]), len(training_data[0][1][0]), 1) #-1 corresponds to how many data samples, and 1 is the number of channels
#X = np.array(X).reshape(-1, len(training_data[0][0]) * len(training_data[0][1][0]), 1) GIVES US 3 DIMENSIONAL X WITH MAIN INPUT FLATTENED

np.save("X_good.npy", X)
np.save("y_good.npy", y)

print ("X, y SPLIT SAVED")

#--------------------CHECKS / WRITING WAV FILES / CLEANED DATA -------------------

print(f"shape of X: {X.shape}")
