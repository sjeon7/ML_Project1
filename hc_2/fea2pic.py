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
start_dir = '../../IAML2017_project_1_2/music/music_training'
pattern   = "*.mp3"
fs = 44100
duration = 29.0
mfcc_dic = {}
split = 10
overlap = 0.5
unit = int(fs * duration / split)
over_unit = int(unit * overlap)
steps = int((split-1)/overlap)+1

def music2mfcc(music, ordinal):
    start, end = ordinal * over_unit , ordinal * over_unit + unit
    stft = np.abs(librosa.stft(x[start:end], n_fft = 2048, hop_length = 512))
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    del stft
    f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    return f

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

    f = []
    for i in range(steps):
        f.append(music2mfcc(x, i))
    f = np.array(f)
    k = path.split("/")[-1].split(".")[0]
    mfcc_dic[k] = f

with open('music/mfcc.pickle', 'wb') as p:
    pickle.dump(mfcc_dic, p)    
