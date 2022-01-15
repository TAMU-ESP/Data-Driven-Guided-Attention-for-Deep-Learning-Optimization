import numpy as np
from scipy import signal as sg
import matplotlib.pyplot as plt
import random

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import sqrt
from collections import defaultdict
import time

from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import sqrt
from collections import defaultdict
import time

from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *

from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import os
import random
from scipy.ndimage import zoom
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import shap
import json
import keras
from keras_self_attention import SeqSelfAttention

from keras.callbacks import History
from keras.preprocessing.sequence import pad_sequences

import keras

import math
import sklearn
from scipy.stats import pearsonr

import glob

#First let's import
import heartpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy
import numpy as np
import glob
from collections import defaultdict

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from sklearn import preprocessing

import biosppy
from scipy.signal import find_peaks, peak_prominences

from scipy.signal import butter, lfilter, freqz
import neurokit2 as nk

from scipy.signal import butter, filtfilt, resample
from BoostedSpringDTW import dynamicEnsemble_BoostedDTW
from sklearn.preprocessing import StandardScaler

import wfdb
import math
from os import path

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler

from fastdtw import fastdtw

def removeZeros(a, n):
 
    # index to store the
    # first non-zero number
    ind = n-1
 
    # traverse in the array
    # and find the first
    # non-zero number
    for i in range(n-1,0,-1):
        if (a[i] != a[-1]):
            ind = i
            break
 
    
    return a[0:ind+1]
    


# path to stored 5-second segments
data_dirs = ["Results/GA_bilstm_attn/personalized/dbp/FinalExp/"]
data_dict = defaultdict(dict)

for data_dir in data_dirs:
    data_dict = defaultdict(dict)
    subjects = os.listdir(data_dir)
    # manually selected and annotated samples
    orig_template = np.load('templates/Sub3.npy', allow_pickle=True)

    orig_template = orig_template[:,1]
    first_grad_template = np.gradient(orig_template)
    second_grad_template = np.gradient(first_grad_template)

    template = np.concatenate((np.reshape(orig_template, (len(orig_template),1)), np.reshape(first_grad_template, (len(first_grad_template),1)), np.reshape(second_grad_template, (len(second_grad_template),1))), axis=1)

    scaler = StandardScaler()
    template = scaler.fit_transform(template)

    dia1 = 0
    ms = 8
    sys = 15
    ip = 34
    dia2 = 69
    for subject in subjects:
        
        print(subject)
        bioz_lists = np.load(data_dir + subject + '/X_test.npy',allow_pickle=True)
        
        new_bioz_list = list()
        for win in bioz_lists:
            zeroedMask = np.zeros((len(win),2))

            ch = 0
            tmp_win = win[:,ch]
            tmp_win = removeZeros(tmp_win,len(tmp_win))
            first_der = np.gradient(tmp_win)
            second_der = np.gradient(first_der)

            tmp_win = scaler.fit_transform(np.reshape(tmp_win, (len(tmp_win),1)))
            first_der = scaler.fit_transform(np.reshape(first_der, (len(first_der),1)))
            second_der = scaler.fit_transform(np.reshape(second_der, (len(second_der),1)))      
            whole_temp = np.concatenate((np.reshape(tmp_win, (len(tmp_win),1)), np.reshape(first_der, (len(first_der),1)), np.reshape(second_der, (len(second_der),1))), axis=1)

            distance1, path1 = fastdtw(first_grad_template, tmp_win, radius=1, dist=euclidean)

            ones_list,dia1_list,ms_list,sys_list,ip_list,dia2_list = list(),list(),list(),list(),list(),list()
            for p in path1:
                if p[0] == dia1:
                  dia1_list.append(p[1])
                elif p[0] == ms:
                  ms_list.append(p[1])
                elif p[0] == sys:
                  sys_list.append(p[1])
                elif p[0] == ip:
                  ip_list.append(p[1])
                elif p[0] == dia2:
                  dia2_list.append(p[1])
            ones_list.append(int(sum(dia1_list)/len(dia1_list)))
            ones_list.append(int(sum(ms_list)/len(ms_list)))
            ones_list.append(int(sum(sys_list)/len(sys_list)))
            ones_list.append(int(sum(ip_list)/len(ip_list)))
            ones_list.append(int(sum(dia2_list)/len(dia2_list)))

            for o in ones_list:
                if o >= len(zeroedMask):
                    continue
                zeroedMask[int(o),ch] = 1
                
            ch = 1
            tmp_win = win[:,ch]
            tmp_win = removeZeros(tmp_win,len(tmp_win))
            first_der = np.gradient(tmp_win)
            second_der = np.gradient(first_der)

            tmp_win = scaler.fit_transform(np.reshape(tmp_win, (len(tmp_win),1)))
            first_der = scaler.fit_transform(np.reshape(first_der, (len(first_der),1)))
            second_der = scaler.fit_transform(np.reshape(second_der, (len(second_der),1)))      
            whole_temp = np.concatenate((np.reshape(tmp_win, (len(tmp_win),1)), np.reshape(first_der, (len(first_der),1)), np.reshape(second_der, (len(second_der),1))), axis=1)

            distance1, path1 = fastdtw(first_grad_template, tmp_win, radius=1, dist=euclidean)

            ones_list,dia1_list,ms_list,sys_list,ip_list,dia2_list = list(),list(),list(),list(),list(),list()
            for p in path1:
                if p[0] == dia1:
                  dia1_list.append(p[1])
                elif p[0] == ms:
                  ms_list.append(p[1])
                elif p[0] == sys:
                  sys_list.append(p[1])
                elif p[0] == ip:
                  ip_list.append(p[1])
                elif p[0] == dia2:
                  dia2_list.append(p[1])
            ones_list.append(int(sum(dia1_list)/len(dia1_list)))
            ones_list.append(int(sum(ms_list)/len(ms_list)))
            ones_list.append(int(sum(sys_list)/len(sys_list)))
            ones_list.append(int(sum(ip_list)/len(ip_list)))
            ones_list.append(int(sum(dia2_list)/len(dia2_list)))

            for o in ones_list:
                if o >= len(zeroedMask):
                    continue
                zeroedMask[int(o),ch] = 1
            new_bioz_list.append(zeroedMask)
        save_path = data_dir + subject + '/' 
        np.save(save_path + 'X_dtw_test',np.asarray(new_bioz_list))
