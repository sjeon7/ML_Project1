##
import os
import pickle
from glob import glob
from multiprocessing import Process, Queue
import librosa
import numpy as np
from tqdm import tqdm
##
musics = []
# change start_dir according to the personal data directories
start_dir = '/home/sung/IAML2017_project_1_2/music/music_training/IAML_music_dataset'
pattern = "*.mp3"
fs = 44100
duration = 29.0

for dir, _, _ in os.walk(start_dir):
    musics.extend(glob(os.path.join(dir, pattern)))
##
def feature_mult(num_start, num_end, q):
    mfcc_dic = {}
    for path in tqdm(musics[num_start:num_end]):
        x, sr = librosa.load(path, sr=None, mono = True, duration=duration)
        x = x.tolist()
        if (len(x) < 1278900):
            print("shorter")
            continue
        else:
            x = x[:1278900]
        x = np.array(x)

        #change code for features desired which is written in features.py
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
        del stft
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        # f.shape would be (20,2498)

        k = path.split("/")[-1].split(".")[0]
        mfcc_dic[k] = f
    q.put(mfcc_dic)
##
if __name__ == '__main__':
    MFCC_DIC_MULT = {}
    # change number of queues, process and change args=(starting number, end number)
    Q1 = Queue()
    Q2 = Queue()
    Q3 = Queue()
    Q4 = Queue()
    Q5 = Queue()
    Q6 = Queue()
    Q7 = Queue()
    Q8 = Queue()

    P1 = Process(target=feature_mult, args=(0, 1000, Q1, ))
    P2 = Process(target=feature_mult, args=(1000, 2000, Q2, ))
    P3 = Process(target=feature_mult, args=(2000, 3000, Q3, ))
    P4 = Process(target=feature_mult, args=(3000, 4000, Q4, ))
    P5 = Process(target=feature_mult, args=(4000, 5000, Q5, ))
    P6 = Process(target=feature_mult, args=(5000, 6000, Q6, ))
    P7 = Process(target=feature_mult, args=(6000, 7000, Q7, ))
    P8 = Process(target=feature_mult, args=(7000, len(musics), Q8, ))

    P1.start()
    P2.start()
    P3.start()
    P4.start()
    P5.start()
    P6.start()
    P7.start()
    P8.start()

    MFCC_DIC_MULT.update(Q1.get())
    MFCC_DIC_MULT.update(Q2.get())
    MFCC_DIC_MULT.update(Q3.get())
    MFCC_DIC_MULT.update(Q4.get())
    MFCC_DIC_MULT.update(Q5.get())
    MFCC_DIC_MULT.update(Q6.get())
    MFCC_DIC_MULT.update(Q7.get())
    MFCC_DIC_MULT.update(Q8.get())

    P1.join()
    P2.join()
    P3.join()
    P4.join()
    P5.join()
    P6.join()
    P7.join()
    P8.join()
    # change name such as feature.pickle
    with open('music/mfcc.pickle', 'wb') as p:
        pickle.dump(MFCC_DIC_MULT, p)
##
