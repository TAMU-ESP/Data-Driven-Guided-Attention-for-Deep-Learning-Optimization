import numpy as np
from scipy import signal 
import matplotlib.pyplot as plt
import random
import os
import sys
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import sqrt
from collections import defaultdict
import time
from tensorflow.keras import backend as K
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from scipy.ndimage import zoom
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import shap
import json
import keras
from keras_self_attention import SeqSelfAttention
from keras.callbacks import History
from keras.preprocessing.sequence import pad_sequences
import glob
#First let's import
import heartpy as hp
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from IPython.core.pylabtools import figsize
from sklearn import preprocessing
import biosppy
from scipy.signal import find_peaks, peak_prominences
from scipy.signal import butter, lfilter, freqz
import neurokit2 as nk
from scipy.signal import butter, filtfilt, resample
import wfdb
from os import path
import sklearn
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

data_directory = ''
dtw_data_directory = ''
subjects = next(os.walk(data_directory))[1]
data_dict = defaultdict(dict)
for subject in subjects:
    bioz_lists = pd.read_pickle(data_directory + subject + "/list_data_time_100.pckl")
    bioz_dtw_lists = np.load(dtw_data_directory + subject + "/list_data_time_100_dtwGuidedPts.npy", allow_pickle=True)
    new_bioz_list,new_bioz_dtw_list = list(),list()
    for raw_b,d in zip(bioz_lists,bioz_dtw_lists):
      b = np.gradient(raw_b,axis=0)
      #b = raw_b
      tmp_zeros = np.zeros((100-len(b),5))
      pad_b = np.concatenate((b,tmp_zeros),axis=0)
      new_bioz_list.append(pad_b)
      tmp_dtw_zeros = np.zeros((100-len(d),4))
      pad_d = np.concatenate((d,tmp_dtw_zeros),axis=0)
      new_bioz_dtw_list.append(pad_d)
    bioz_lists = np.asarray(new_bioz_list)
    bioz_dtw_lists = np.asarray(new_bioz_dtw_list)
    bioz_labels = pd.read_pickle(data_directory + subject + "/labels_100.pckl")
    bioz_labels = bioz_labels.values
    data_dict[subject]['X_data'] = bioz_lists[:,:,0:2]
    data_dict[subject]['X_dtw_data'] = bioz_dtw_lists[:,:,0:2]
    data_dict[subject]['BP_data'] = bioz_labels
    
for subject in subjects:
    print(subject)
    print('Loading data...')
    
    save_directory = "Results/bilstm_attn/personalized/dbp/e72/"
    
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    X_data = np.asarray(data_dict[subject]['X_data'])
    X_dtw_data = np.asarray(data_dict[subject]['X_dtw_data'])
    sbp_data = np.asarray(data_dict[subject]['BP_data'])[:,1]
    dbp_data = np.asarray(data_dict[subject]['BP_data'])[:,0]
    indices = np.arange(len(X_data))
    np.random.shuffle(indices)
    X_data = X_data[indices]
    X_dtw_data = X_dtw_data[indices]
    sbp_data = sbp_data[indices]
    dbp_data = dbp_data[indices]
    bp_data = np.concatenate((np.reshape(dbp_data,(len(dbp_data),1)),np.reshape(sbp_data,(len(sbp_data),1))),axis=1)
    
    training_size = 0.8
    x_train = X_data[0:int(len(X_data)*training_size)]     
    x_dtw_train = X_dtw_data[0:int(len(X_dtw_data)*training_size)]
    sbp_train = sbp_data[0:int(len(sbp_data)*training_size)]
    dbp_train = dbp_data[0:int(len(dbp_data)*training_size)]
    bp_train = bp_data[0:int(len(bp_data)*training_size)]
    
    x_test = X_data[int(len(X_data)*training_size):]     
    x_dtw_test = X_dtw_data[int(len(X_dtw_data)*training_size):]
    sbp_test = sbp_data[int(len(sbp_data)*training_size):]
    dbp_test = dbp_data[int(len(dbp_data)*training_size):]
    bp_test = bp_data[int(len(bp_data)*training_size):]


    X_train = x_train
    X_test = x_test
    X_dtw_train = x_dtw_train
    X_dtw_test = x_dtw_test

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Channel 1 inputs
    ch1_X_train = X_train[:,:,0]
    ch1_X_test = X_test[:,:,0]
    ch1_X_dtw_train = x_dtw_train[:,:,0]
    ch1_X_dtw_test = x_dtw_test[:,:,0]


    # Channel 2 inputs
    ch2_X_train = X_train[:,:,1]
    ch2_X_test = X_test[:,:,1]
    ch2_X_dtw_train = x_dtw_train[:,:,1]
    ch2_X_dtw_test = x_dtw_test[:,:,1]
    
    ch1_X_train = np.reshape(ch1_X_train,(len(ch1_X_train),len(ch1_X_train[0]),1))
    ch2_X_train = np.reshape(ch2_X_train,(len(ch2_X_train),len(ch2_X_train[0]),1))
    ch1_X_test = np.reshape(ch1_X_test,(len(ch1_X_test),len(ch1_X_test[0]),1))
    ch2_X_test = np.reshape(ch2_X_test,(len(ch2_X_test),len(ch2_X_test[0]),1))

    if not os.path.exists(save_directory + subject + "/models/"):
        os.makedirs(save_directory + subject + "/models/")

    print('Init model...')
    

    wave_in = tf.keras.Input(shape=X_train[0].shape)
    l1_h1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(wave_in)
    a1_h1 = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention')(l1_h1)
    f1_h1 = tf.keras.layers.Flatten()(a1_h1)
    #dbp_out = tf.keras.layers.Dense(1)(f1_h1)
    bp_out = tf.keras.layers.Dense(1)(f1_h1)
    
    model = keras.Model(
        inputs=[wave_in],
        outputs=[bp_out],
    )
    opt = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(
        optimizer=opt,
        loss=[
            'mse'
        ]
    )
    
    model.summary()

    history = History()
    # checkpoint
    if not os.path.exists(save_directory + subject + '/models/'):
        os.makedirs(save_directory + subject + '/models/')
    weights_filepath = save_directory + subject + '/models/best_weights_exp1.hdf5'
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
    callbacks_list = [checkpoint,history]

    model.fit(X_train, dbp_train, validation_split=0.1, shuffle=True, epochs=50, batch_size=16, callbacks=callbacks_list)

    model.load_weights(weights_filepath)
    #model.save_weights(weights_filepath)

    preds = model.predict(X_test)

    np.save(save_directory + subject + '/X_test',X_test)
    np.save(save_directory + subject + '/BP_test',dbp_test)
    np.save(save_directory + subject + '/preds',preds)
    
    K.clear_session()
    
    print('Done!')
    
    save_directory = "Results/GA_bilstm_attn/personalized/dbp/e72/"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    wave_in = tf.keras.Input(shape=X_train[0].shape)
    
    l1_h1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(wave_in)
    a1_h1 = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention')(l1_h1)
    f1_h1 = tf.keras.layers.Flatten()(a1_h1)
    ch1_pts = tf.keras.layers.Dense(len(X_train[0]),activation='sigmoid')(f1_h1)
    ch2_pts = tf.keras.layers.Dense(len(X_train[0]),activation='sigmoid')(f1_h1)
    bp_out = tf.keras.layers.Dense(1)(f1_h1)
    
    
    model = keras.Model(
        inputs=[wave_in],
        outputs=[ch1_pts,ch2_pts,bp_out],
    )
    opt = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(
        optimizer=opt,
        loss=[
            tf.keras.losses.BinaryCrossentropy(),
            tf.keras.losses.BinaryCrossentropy(),
            'mse'
        ],
        loss_weights=[
            1.0,
            1.0,
            0.05
        ]
    )
    
    model.summary()

    history = History()
    # checkpoint
    if not os.path.exists(save_directory + subject + '/models/'):
        os.makedirs(save_directory + subject + '/models/')
    weights_filepath = save_directory + subject + '/models/best_weights_exp1.hdf5'
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_dense_2_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
    callbacks_list = [checkpoint,history]

    model.fit(X_train, [ch1_X_dtw_train,ch2_X_dtw_train,dbp_train], validation_split=0.1, shuffle=True, epochs=50, batch_size=16, callbacks=callbacks_list)

    model.load_weights(weights_filepath)
    #model.save_weights(weights_filepath)

    preds = model.predict(X_test)

    np.save(save_directory + subject + '/X_test',X_test)
    np.save(save_directory + subject + '/BP_test',dbp_test)
    np.save(save_directory + subject + '/preds_0',preds[0])
    np.save(save_directory + subject + '/preds_1',preds[1])
    np.save(save_directory + subject + '/preds_2',preds[2])
    
    K.clear_session()
    
    print('Done!')
    
    
