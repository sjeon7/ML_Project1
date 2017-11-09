##
import os
import numpy as np
from scipy import stats
import pandas as pd
import librosa
import argparse
from tqdm import *
import pickle

def compute_features(config):
    df=pd.read_csv('dataset/track_metadata.csv')
    
    if config.training == 'True':
        metadata_df = df[df['set_split']=='training']
    else:
        metadata_df = df[df['set_split']=='validation']

#    pointer = 0
#    batch_size = 32
#    num_batch = int(len(metadata_df) / batch_size)
#    pointer = (pointer + 1) % num_batch
#    start_pos = pointer * batch_size
#    meta_df = metadata_df.iloc[start_pos:(start_pos+batch_size)]        
        
    tids = metadata_df['track_id'].values
    threshold = 1278900

    successful_tids = []
    successful_features = []
    
    print('Feature: ' + config.feature + ', Training: ' + str(config.training))
    
#    tbar = tqdm(tids)
    for tid in tqdm(tids, ncols=80):
#        tbar.set_description("Processing %d" % tid)
        try:
            filepath = get_audio_path('music/music_training', tid)
            x, sr = librosa.load(filepath, sr=None, mono=True, duration=29.0)  # kaiser_fast
            x = x.tolist()
            if(len(x)<threshold):
                raise ValueError('song length is shorter than threshold')
            else:
                x = x[:1278900]
            x = np.array(x)

            
            if config.feature == 'zero_crossing_rate':
                # returns (1,t)
                f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)                
            
            elif config.feature == 'chroma_cqt':
                cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                         n_bins=7 * 12, tuning=None))
                assert cqt.shape[0] == 7 * 12
                assert np.ceil(len(x) / 512) <= cqt.shape[1] <= np.ceil(len(x) / 512) + 1
                # returns (n_chroma, t)
                f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)

            elif config.feature == 'chroma_cens':
                cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                         n_bins=7 * 12, tuning=None))
                assert cqt.shape[0] == 7 * 12
                assert np.ceil(len(x) / 512) <= cqt.shape[1] <= np.ceil(len(x) / 512) + 1                
                # returns (n_chroma, t)
                f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)

            elif config.feature == 'chroma_stft':                
                stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
                assert stft.shape[0] == 1 + 2048 // 2
                assert np.ceil(len(x) / 512) <= stft.shape[1] <= np.ceil(len(x) / 512) + 1
                del x
                # returns (n_chroma, t)
                f = librosa.feature.chroma_stft(S=stft ** 2, n_chroma=12)                

            elif config.feature == 'rmse':                
                stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
                assert stft.shape[0] == 1 + 2048 // 2
                assert np.ceil(len(x) / 512) <= stft.shape[1] <= np.ceil(len(x) / 512) + 1
                del x
                # returns (1,t)
                f = librosa.feature.rmse(S=stft)     

            elif config.feature == 'spectral_centroid':                
                stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
                assert stft.shape[0] == 1 + 2048 // 2
                assert np.ceil(len(x) / 512) <= stft.shape[1] <= np.ceil(len(x) / 512) + 1
                del x
                # returns (1,t)
                f = librosa.feature.spectral_centroid(S=stft)            
            
            elif config.feature == 'spectral_bandwidth':                
                stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
                assert stft.shape[0] == 1 + 2048 // 2
                assert np.ceil(len(x) / 512) <= stft.shape[1] <= np.ceil(len(x) / 512) + 1
                del x
                # returns (1,t)
                f = librosa.feature.spectral_bandwidth(S=stft)
            
            elif config.feature == 'spectral_contrast':                
                stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
                assert stft.shape[0] == 1 + 2048 // 2
                assert np.ceil(len(x) / 512) <= stft.shape[1] <= np.ceil(len(x) / 512) + 1
                del x
                # returns (n_bands+1, t)
                f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
            
            elif config.feature == 'spectral_rolloff':
                stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
                assert stft.shape[0] == 1 + 2048 // 2
                assert np.ceil(len(x) / 512) <= stft.shape[1] <= np.ceil(len(x) / 512) + 1
                del x
                # returns (1,t)
                f = librosa.feature.spectral_rolloff(S=stft)
                
            else: # mfcc
                stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
                assert stft.shape[0] == 1 + 2048 // 2
                assert np.ceil(len(x) / 512) <= stft.shape[1] <= np.ceil(len(x) / 512) + 1
                del x
                # returns (n_mfcc, t)
                mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
                del stft
                f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
            
            successful_tids.append(tid)
            successful_features.append(f.tolist())
            
        except Exception as e:
            print('{}: {}'.format(tid, repr(e)))
            return tid, 0

    sf = np.array(successful_features)
    sf = np.reshape(sf, (sf.shape[0], sf.shape[1]*sf.shape[2]))

    successful = np.c_[np.array(successful_tids), sf]
    successful = pd.DataFrame(successful)

    if config.training == 'True':
        successful.to_csv('dataset/' + config.feature + '_training.csv', index=False, header=False)
    else:
        successful.to_csv('dataset/' + config.feature + '_validation.csv', index=False, header=False)

def get_audio_path(audio_dir, track_id):
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir + '/' + tid_str[:3] + '/' + tid_str + '.mp3')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str, choices=['zero_crossing_rate', 
                                                        'chroma_cqt', 
                                                        'chroma_cens', 
                                                        'chroma_stft', 
                                                        'rmse', 
                                                        'spectral_centroid', 
                                                        'spectral_bandwidth', 
                                                        'spectral_contrast', 
                                                        'spectral_rolloff', 
                                                        'mfcc'],
                       default='mfcc', help='feature to compute')
    parser.add_argument('--training', type=str, choices=['True', 'False'], default=True, 
                        help='whether it is training or validation')
    
    config = parser.parse_args()
    compute_features(config)
    
if __name__ == '__main__':
    main()
