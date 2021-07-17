import os
import librosa
import librosa.display
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn import preprocessing


warnings.filterwarnings('ignore')

df = pd.read_csv('public_dataset/metadata_compiled.csv')

covid_list = []
for uuid, status in zip(df['uuid'], df['status']):
  if status == 'COVID-19':
    covid_list.append(uuid)

healthy_list = []
for uuid, status in zip(df['uuid'], df['status']):
  if status != 'healthy':
    healthy_list.append(uuid)
  if len(healthy_list) == 1500:
    break

try:
    os.makedirs('CoughVID')
    os.makedirs('CoughVID/Images')
    os.makedirs('CoughVID/Images/covid')
    os.makedirs('CoughVID/Images/healthy')
    os.makedirs('CoughVID/MFCC')
    os.makedirs('CoughVID/MFCC/covid')
    os.makedirs('CoughVID/MFCC/healthy')

except:
    pass

def extract_features(id_, label):
  try:
    audio_clip, sample_rate = librosa.load(os.path.join('public_dataset', id_ + '.webm'), sr=None)
  except:
    audio_clip, sample_rate = librosa.load(os.path.join('public_dataset', id_ + '.ogg'), sr=None)
  
  spec = librosa.stft(audio_clip)
  spec_mag, _ = librosa.magphase(spec)
  mel_spec = librosa.feature.melspectrogram(S=spec_mag, sr=sample_rate)
  log_spec = librosa.amplitude_to_db(mel_spec, ref=np.min)
  fig,ax = plt.subplots(1)
  fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
  ax.axis('tight')
  ax.axis('off')
  img = librosa.display.specshow(log_spec, sr=sample_rate)
  img_path = os.path.join('CoughVID/Images/', label, str(id_) + '.png')
  fig.savefig(img_path)
  plt.close(fig)
  mfcc = librosa.feature.mfcc(audio_clip, sr=sample_rate)
  mfcc = preprocessing.scale(mfcc, axis=1)
  np.save(os.path.join('CoughVID/MFCC/', label, str(id_) + '.npy'), mfcc)

for i, id_ in tqdm(enumerate(covid_list)):
  extract_features(id_, 'covid')

for i, id_ in tqdm(enumerate(healthy_list)):
  extract_features(id_, 'healthy')