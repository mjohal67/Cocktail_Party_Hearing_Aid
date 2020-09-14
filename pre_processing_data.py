import numpy as np
import librosa
from librosa import display
import os
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import scipy.stats as stats
import tqdm

#implement progress bars for sections (and figure out how to label them)
#could figure out how to batch the data depending on how long this takes

#directories

DATADIR = "training_speechwavs"
FOLDERS = ["Speech", "Noise_Wavs"]

#data storage
to_noise = []

unmixed_speech_files = []
unmixed_noise_files = []

normed_speech_files = []
normed_noise_files = []

IRMs = []
mixed_files = []

training_data = []

#audio params

clip_length = 64000 #4 seconds
n_fft = 512
hop_length = int(n_fft//2)
samplerate = 16000

def create_training_data():
    global unmixed_noise_files
    global unmixed_speech_files
    global training_data
    global to_noise
    global IRMs
    global stft_mix
    global clip_length
    global n_fft
    global hop_length
    global samplerate

    samples = 0

    for folder in FOLDERS:
        path = os.path.join(DATADIR, folder)
        for file in os.listdir(path):
            try:
                signal, _ = librosa.load(os.path.join(path, file), sr=samplerate)

                if folder == "Speech":
                    #for i in range(0, len(signal), clip_length):
                    #    print (f"i:" + str(i))
                    #    chopped_signal = signal[i:i+clip_length]
                    #    print (f"length of chopped_signal: {len(chopped_signal)}")
                    #    if len(chopped_signal) == int(clip_length):
                    #        unmixed_speech_files.append(np.array(chopped_signal))


                    signal = signal[0:clip_length]
                    if len(signal) == int(clip_length):
                        unmixed_speech_files.append(np.array(signal))
                        samples += 1
                        print("Samples processed: ", int(samples))

                elif folder == "Noise_Wavs":

                    '''only use this if we have noise files 12 seconds or longer
                    for i in range(0, len(signal), clip_length):
                        sliced_signal = signal[i:i+clip_length]
                        #print (len(sliced_signal))
                        if len(sliced_signal) == int(clip_length):
                            to_noise.append(np.array(sliced_signal))
                    '''

                    signal = signal[0:clip_length]
                    if len(signal) == int(clip_length):
                        to_noise.append(np.array(signal))
                        samples += 1
                        print("Samples processed: ", int(samples))

            except Exception as e:
                print (e)

    for i in range(0, len(to_noise), 4):
        if (int(i)) <= int(len(to_noise)-4): #as long as there's still 4 more to be able to be added in a row
            added = to_noise[i] + to_noise[i+1] + to_noise[i+2] + to_noise[i+3] #add three more to that one, so that we now have a mix of four total files. After this, it'll skip four ahead to the next element (ex. if it started at 0, we add 1, 2, 3, and then the first for loop would skip to the 4th element, and add three more from there)
            unmixed_noise_files.append(added)
#    else:
#        pass


    unmixed_speech_files = np.array(unmixed_speech_files)
    if len(unmixed_noise_files) > len(unmixed_speech_files):
        unmixed_noise_files = np.array(unmixed_noise_files[0:len(unmixed_speech_files)])
    elif len(unmixed_speech_files) > len(unmixed_noise_files):
        unmixed_speech_files = np.array(unmixed_speech_files[0:len(unmixed_noise_files)])

    #print (f"Items in unmixed SPEECH files: {len(unmixed_speech_files)}")
    #print (f"Items in unmixed NOISE files: {len(unmixed_noise_files)}")


    for (speech_data, noise_data) in zip(unmixed_speech_files, unmixed_noise_files):
        try:
            speech_data = speech_data/np.linalg.norm(speech_data) #l2 norm takes the euclidian distance of the vector (so, the hypotenuse of a triangle). This is the "greatest value", and then, we divide everything by that. So, our hypotenuse becomes one, and everything else becomes less than one (just like the unit circle)
            noise_data = noise_data/np.linalg.norm(noise_data)

            max_speech = max(abs(speech_data)) #these two lines should do the same thing as the matlab one-liner
            max_noise = max(abs(noise_data))
            ampAdj = max((max_speech, max_noise)) #finding the max amplitude out of each sample

            speech_data = speech_data/ampAdj #dividing by this so that they're each the same volume
            noise_data = noise_data/ampAdj

            mixed_data = speech_data + noise_data
            mixed_data = mixed_data / max(mixed_data) #how is this and finding the l2 norm different?
            mixed_data = np.array(mixed_data)

            stft_speech = np.array(librosa.core.stft(y=speech_data, n_fft=n_fft, hop_length=hop_length)) #could lowpass our signal so that our fmax is 8000Hz
            stft_noise = np.array(librosa.core.stft(y=noise_data, n_fft=n_fft, hop_length=hop_length)) #we need to keep phase information for reconstructability, so we EITHER just feed this into our neural network (might have to use a complex neural network??), OR we keep phase separatly, and then multiply it in either before or after (I'm pretty sure it's before) we multiply the mix by our irm
            stft_mix = np.array(librosa.core.stft(y=mixed_data, n_fft=n_fft, hop_length=hop_length))


            #type of standardization only works if data is normal (guassian)
            #if we wanted to check to see if our data is guassian with a Q-Q plot
            #stats.probplot(mixed_data, dist="norm", plot=plt)
            #plt.show()

            '''
            1. shuffle and np.save training data with the mix file (phase and magnitude all together), and it's respective IRM
            2. at the bottom of this file, load the training data, isolate magnitude and phase - EITHER keep both in one vector separated (both real, ex. [[magnitude_info], [phase_info]]    ), or JUST KEEP THE MAGNITUDE (start with this!! --> this would be our X list [our feature list]) - and with both options, the corresponding label would be the IRM (this would be our y list [our label list])
            3. in the for loop, before we append features and labels (in this case, magnitude and IRM) to their respective lists, normalize/standardize, then SAVE THE X AND Y LISTS! (print data once standardized to make sure it is properly standardized)
            '''

            irm = abs(stft_speech) / (abs(stft_noise) + abs(stft_speech)) #don't take into account phase information here, just trying to get ratio of speech to mix, and we don't want to add any extra phase into the cleaned mix when we multiply IRM in
            #IRMs.append(irm)

            #we wouldn't want to standardize our irm because the irm itself is a ratio (some value between 0 and 1) --> standardization of this would make values between -1 and 1, which wuold throw off our masking

            training_data.append([stft_mix, irm]) #non-normalized mix stft with it's corresponding IRM
        except Exception as f:
            print (f)

    training_data = np.asarray(training_data, dtype=np.complex64) #I think float 32 should be okay type for this data as long as tensorflow supports it
    np.random.shuffle(training_data)
#Batch data into segments that are each saved to separate .npy files, and once all files have been created, load all these files and append their contents together

    np.save("mix_irm_training_data_complex_good.npy", training_data)
    print ("TRAINING DATA SAVED")


create_training_data()

print (" ")
print (f"Items in TRAINING data: {len(training_data)}")
print(f"shape of each mix_stft: {training_data[0][0].shape}")
print(f"shape of each irm: {training_data[0][1].shape}")
print(f"shape of training_data: {training_data.shape}")
#print(f"number of irms: {len(IRMs)}")
