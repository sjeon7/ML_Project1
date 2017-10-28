import os, sys
import librosa
import pandas as pd
import numpy as np
from dataloader import DataLoader
from time import strftime, localtime, time
from tqdm import tqdm
import pickle
from glob import glob

musics = []
start_dir = 'music/music_training'
pattern   = "*.mp3"
fs = 44100
duration = 29.0
mfcc_dic = {}

for dir, _, _ in os.walk(start_dir):
    musics.extend(glob(os.path.join(dir, pattern))) 

for path in tqdm(musics):
    x, sr = librosa.load(path, sr=None, mono = True, duration=duration)
    x = x.tolist()
    if (len(x) < 1278900):
        print("shorter")
        continue
    else:
        x = x[:1278900]
    x = np.array(x)

    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
    del stft
    f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    # f.shape would be (20,2498)

    k = path.split("/")[-1].split(".")[0]
    mfcc_dic[k] = f

with open('music/mfcc.pickle', 'wb') as p:
    pickle.dump(mfcc_dic, p)    
