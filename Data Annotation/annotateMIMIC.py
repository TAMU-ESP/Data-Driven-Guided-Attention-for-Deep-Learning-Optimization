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

data_dirs = ["Results/GA_cnn_bilstm_attn/personalized/dbp/FinalExp/"]
data_dict = defaultdict(dict)

for data_dir in data_dirs:
    data_dict = defaultdict(dict)
    subjects = os.listdir(data_dir)
    
    for subject in subjects:
        #if subject == 'p087474-2105-02-12-10-27':
        #    continue
        #if os.path.exists(data_dir + subject + '/X_dtw_test.npy'):
        #    continue
        print(subject)
        data_dict[subject]['X_data'] = np.load(data_dir + subject + '/X_test.npy',allow_pickle=True)
        
        x_dtw_list = []
        count_x = 0
        num_samples = len(data_dict[subject]['X_data'])
        for x in data_dict[subject]['X_data']:
            ### Fill template set ###
            template_set = defaultdict(dict)
            temp_count = 0

            temp1 = np.load('templates/mimic_revised_template.npy', allow_pickle=True)

            scaler = StandardScaler()
            temp1 = scaler.fit_transform(temp1)

            template_set[temp_count]['seg'] = temp1
            template_set[temp_count]['dicr'] = 10
            template_set[temp_count]['sys'] = 18
            template_set[temp_count]['dia'] = 50
            template_set[temp_count]['frequency'] = 0
            template_set[temp_count]['template_ID'] = 0

            save_path = data_dir + subject + '/' 

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            #np.save(save_path + 'templates/' + str(template_set[temp_count]['template_ID']) + '.npy',temp1)
            temp_count += 1

            bvp = pd.DataFrame(columns={'BVP'})
            bvp['BVP'] = x
            bvp['BVP_1'] = np.gradient(bvp['BVP'].values)
            bvp['BVP_2'] = np.gradient(bvp['BVP_1'].values)

            t_bvp = np.arange(0.008, 0.008 * (625 + 1), 0.008)

            bvp.set_index(keys=pd.Index(t_bvp), inplace=True)

            rs_DTW = dynamicEnsemble_BoostedDTW(template_set=template_set, save_path='')
            res = rs_DTW.extractFeatures(bvp=bvp)

            try:
                start_pts = np.asarray(res['step_start'].values)
            except:
                start_pts = np.asarray([])
            try:
                end_pts = np.asarray(res['step_end'].values)
            except:
                end_pts = np.asarray([])
            try:
                ms_pts = np.add(start_pts,res['dicr_step'].values.astype(int))
            except:
                ms_pts = np.asarray([])
            try:
                sys_pts = np.add(start_pts,res['sys_step'].values.astype(int))
            except:
                sys_pts = np.asarray([])
            try:
                dia_pts = np.add(start_pts,res['dia_step'].values.astype(int))
            except:
                dia_pts = np.asarray([])
            all_pts = start_pts.tolist() + end_pts.tolist() + ms_pts.tolist() + sys_pts.tolist() + dia_pts.tolist()

            new_all_pts = []
            for p in all_pts:
                if (p < 0) or (p >= 625):
                    continue
                new_all_pts.append(p)

            ppg_pts_mask = np.zeros(625)

            ppg_pts_mask[new_all_pts] = 1

            x_dtw_list.append(ppg_pts_mask)

            count_x += 1
            print(count_x)
            if count_x == round(num_samples/4):
                print('Completed 25%')
            if count_x == round(num_samples/2):
                print('Completed 50%')
            if count_x == round(3*(num_samples/4)):
                print('Completed 75%')
            if count_x == num_samples:
                print('Completed 100%')

            #if count_x == 200:
            #    break
        np.save(save_path + 'X_dtw_data',np.asarray(x_dtw_list))
        


        
        
