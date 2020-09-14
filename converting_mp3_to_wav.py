import numpy as np
import librosa
from librosa import display
import os
from pydub import AudioSegment
from tqdm import tqdm

speech_path = "training_speechwavs/Noise"
output_speech_path = "training_speechwavs/Noise_Wavs"

for file in tqdm(os.listdir(speech_path)):
    try:
#        og_file = AudioSegment.from_mp3(os.path.join(speech_path, file))
#        og_file.export(os.path.join(output_speech_path, file[:-4]+".wav"), format="wav")
    except Exception as e:
        print (e)
