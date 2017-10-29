import os
import random
import argparse

features = [#'mfcc', 
            'chroma_cqt', 
            'chroma_cens', 
            'chroma_stft', 
            'rmse', 
            'spectral_centroid', 
            'spectral_bandwidth', 
            'spectral_contrast', 
            'spectral_rolloff',
            'zero_crossing_rate']
training = ['True', 'False']

for feature in features:
    for train in training:
        os.system('python features.py --feature {0} --training {1}'.format(feature, train))