#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import pandas as pd


# In[11]:


def wav_files_to_csv(directory_list, dest_path):
    classes = {"yes": 0, "no": 1, "up": 2, "down": 3, "left": 4, "right": 5, "on": 6,
               "off": 7, "stop": 8, "go": 9, "unknown": 10, "silence": 11}
    columns = ["labels"]
    for i in range(9159):
        columns.append("p" + str(i))
    df = pd.DataFrame(columns=columns)
    path = "../resources/train/audio/"
    
    dir_count = 1
    row = 0
    for directory in directory_list:
        file_names = next(os.walk(os.path.join(path, directory)))[2]
        random.shuffle(file_names)
        #file_names = file_names[:800] # grab a sample of 800 for every category
        file_count = 0
        for file in file_names:
            audio = os.path.join(path, directory, file)
            sample_rate, samples = wavfile.read(audio)
            if (samples.shape[0] < 16000):
                samples = np.append(samples, np.zeros(16000 - samples.shape[0]))
            freq, times, spectrogram = signal.spectrogram(samples, sample_rate)
            spectrogram = spectrogram.flatten()
            spectrogram = np.where(spectrogram == 0, 0, np.log(spectrogram))
            spectrogram = np.insert(spectrogram, 0, classes[directory]) # insert label
            df.loc[row] = spectrogram
            row += 1
            file_count += 1
            if (file_count % 100 == 0):
                print("file: {} dir: {} of {}".format(file_count, dir_count, len(directory_list)))
        dir_count += 1
    print("converting to csv...")
    df = df.astype({"labels": int})
    df = df.sample(frac=1) #shuffle rows
    df.to_csv(path_or_buf=dest_path, index=False)
    return (df)


# In[13]:


dir_list = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
#dir_list = ["yes", "no"]
yes_no = wav_files_to_csv(dir_list, "../resources/train_csv/full_dataset.csv")


# In[ ]:


#directory = "resources/train/audio/one/"
#file_names = next(os.walk(directory))[2]


# In[ ]:


def display_random(directory, file_names):
    audio = os.path.join(directory, random.choice(file_names))
    sample_rate, sample = wavfile.read(audio)
    freq, times, spectogram = signal.spectrogram(sample, sample_rate)
    print("sample =", sample.shape)
    print("sample_rate =", sample_rate)
    print("audio_path =", audio)
    print("spectogram =", spectogram.shape)
    f, ax = plt.subplots(figsize=(20, 15))
    plt.pcolormesh(times, freq, spectogram)
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time in [sec]")
    plt.show()
    plt.specgram(sample, Fs=sample_rate);


# In[ ]:




