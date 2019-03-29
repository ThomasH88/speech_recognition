#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import pandas as pd
import os
import sys


# In[ ]:


def wav_files_to_csv(dest_path, start, end):
    path = "../resources/test/audio/"
    file_names = next(os.walk(os.path.join(path)))[2]
    file_names.sort()
    file_names = file_names[start:end]
    arr_1 = np.zeros(shape=((end - start) // 2, 9159))
    arr_2 = np.zeros(shape=((end - start) // 2, 9159))
    file_count = 0
    row = 0
    for file in file_names:
        audio = os.path.join(path, file)
        sample_rate, samples = wavfile.read(audio)
        if (samples.shape[0] < 16000):
            samples = np.append(samples, np.zeros(16000 - samples.shape[0]))
        freq, times, spectrogram = signal.spectrogram(samples, sample_rate)
        spectrogram = spectrogram.flatten()
        spectrogram = np.where(spectrogram == 0, 0, np.log(spectrogram))
        if (file_count == arr_1.shape[0]):
            row = 0
        if (file_count >= arr_1.shape[0]):
            arr_2[row] = spectrogram
        else:
            arr_1[row] = spectrogram
        row += 1
        file_count += 1
        if (file_count % 200 == 0):
            print("({}-{}) file: {} of {}".format(start, end, file_count, end - start))
    print("converting to csv...")
    columns = []
    for i in range(9159):
        columns.append("p" + str(i))
    df = pd.DataFrame(data=arr_1, columns=columns)
    df["fname"] = file_names[:arr_1.shape[0]]
    df = df.sample(frac=1) #shuffle rows
    df.to_csv(path_or_buf=dest_path + "test_{}_{}_1.csv".format(start, end), index=False)
    
    del df
    df = pd.DataFrame(data=arr_2, columns=columns)
    df["fname"] = file_names[arr_1.shape[0]:]
    df = df.sample(frac=1) #shuffle rows
    df.to_csv(path_or_buf=dest_path + "test_{}_{}_2.csv".format(start, end), index=False)


# In[ ]:


wav_files_to_csv("../resources/test_csv/", int(sys.argv[1]), int(sys.argv[2]))
#wav_files_to_csv("../resources/test_csv/", 0, 10000)


# In[ ]:


# test = pd.read_csv("../resources/test_csv/test_0_10000_1.csv")
# test2 = pd.read_csv("../resources/test_csv/test_0_10000_2.csv")


# In[ ]:


# test = test.sort_values("fname")
# test2 = test2.sort_values("fname")
# test = test.reset_index(drop=True)
# test2 = test2.reset_index(drop=True)
# fnames = test["fname"]
# fnames2 = test2["fname"]
# test2 = test2.drop(labels="fname", axis='columns')
# test = test.drop(labels="fname", axis='columns')


# In[ ]:


# nb_1 = 3567
# print(fnames[nb_1])
# plt.pcolormesh(test.values[nb_1].reshape(129, 71));


# In[ ]:


# print(fnames2[nb_1])
# plt.pcolormesh(test2.values[nb_1].reshape(129, 71));

