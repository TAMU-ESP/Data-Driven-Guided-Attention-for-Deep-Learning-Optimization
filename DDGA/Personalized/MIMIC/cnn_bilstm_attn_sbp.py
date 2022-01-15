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

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

data_dir = "../Annotated_Dataset/"
subjects = os.listdir(data_dir)
data_dict = defaultdict(dict)
for subject in subjects:

    print(subject)  
    data_dict[subject]['X_data'] = np.load(data_dir + subject + '/X_data.npy',allow_pickle=True)
    data_dict[subject]['X_dtw_data'] = np.load(data_dir + subject + '/X_dtw_data.npy',allow_pickle=True)
    data_dict[subject]['SBP_data'] = np.load(data_dir + subject + '/SBP_data.npy',allow_pickle=True)
    data_dict[subject]['DBP_data'] = np.load(data_dir + subject + '/DBP_data.npy',allow_pickle=True)
    print(data_dict[subject]['X_data'].shape)
    
for subject in subjects:

    print(subject)
    print('Loading data...')
    
    save_directory = "Results/cnn_bilstm_attn/personalized/sbp/e70/"
    
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    X_data = np.asarray(data_dict[subject]['X_data'])
    X_dtw_data = np.asarray(data_dict[subject]['X_dtw_data'])
    sbp_data = np.asarray(data_dict[subject]['SBP_data'])
    dbp_data = np.asarray(data_dict[subject]['DBP_data'])
    indices = np.arange(len(X_data))
    np.random.shuffle(indices)
    X_data = X_data[indices]
    X_dtw_data = X_dtw_data[indices]
    sbp_data = sbp_data[indices]
    dbp_data = dbp_data[indices]
    
    training_size = 0.8
    x_train = X_data[0:int(len(X_data)*training_size)]     
    x_dtw_train = X_dtw_data[0:int(len(X_dtw_data)*training_size)]
    sbp_train = sbp_data[0:int(len(sbp_data)*training_size)]
    dbp_train = dbp_data[0:int(len(dbp_data)*training_size)]
    
    x_test = X_data[int(len(X_data)*training_size):]     
    x_dtw_test = X_dtw_data[int(len(X_dtw_data)*training_size):]
    sbp_test = sbp_data[int(len(sbp_data)*training_size):]
    dbp_test = dbp_data[int(len(dbp_data)*training_size):]


    X_train = x_train
    X_test = x_test
    X_dtw_train = x_dtw_train
    X_dtw_test = x_dtw_test

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    ch1_X_train = X_train
    ch1_X_test = X_test
    ch1_X_dtw_train = x_dtw_train
    ch1_X_dtw_test = x_dtw_test


 
    ch1_X_train = np.reshape(ch1_X_train,(len(ch1_X_train),len(ch1_X_train[0]),1))
    ch1_X_test = np.reshape(ch1_X_test,(len(ch1_X_test),len(ch1_X_test[0]),1))

    if not os.path.exists(save_directory + subject + "/models/"):
        os.makedirs(save_directory + subject + "/models/")

    print('Init model...')
    

    wave_in = tf.keras.Input(shape=ch1_X_train[0].shape)
    cn1_h1 = tf.keras.layers.Conv1D(32, 3, activation=None)(wave_in)
    bn1_h1 = tf.keras.layers.BatchNormalization()(cn1_h1)
    re1_h1 = tf.keras.layers.ReLU()(bn1_h1)
    mp1_h1 = tf.keras.layers.MaxPooling1D(pool_size=3,strides=1, padding='valid')(re1_h1)
    cn2_h2 = tf.keras.layers.Conv1D(64, 3, activation=None)(mp1_h1)
    bn2_h2 = tf.keras.layers.BatchNormalization()(cn2_h2)
    re2_h2 = tf.keras.layers.ReLU()(bn2_h2)
    mp2_h2 = tf.keras.layers.MaxPooling1D(pool_size=3,strides=1, padding='valid')(re2_h2)
    cn3_h3 = tf.keras.layers.Conv1D(128, 3, activation=None)(mp2_h2)
    bn3_h3 = tf.keras.layers.BatchNormalization()(cn3_h3)
    re3_h3 = tf.keras.layers.ReLU()(bn3_h3)

    mp3_h3 = tf.keras.layers.MaxPooling1D(pool_size=3,strides=1, padding='valid')(re3_h3)
    
    l1_h1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(mp3_h3)
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

    model.fit(X_train, sbp_train, validation_split=0.1, shuffle=True, epochs=50, batch_size=128, callbacks=callbacks_list)

    model.load_weights(weights_filepath)
    #model.save_weights(weights_filepath)

    preds = model.predict(X_test)

    np.save(save_directory + subject + '/X_test',X_test)
    np.save(save_directory + subject + '/BP_test',sbp_test)
    np.save(save_directory + subject + '/preds',preds)
    
    K.clear_session()
    
    print('Done!')
    
    save_directory = "Results/GA_cnn_bilstm_attn/personalized/sbp/e70/"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    wave_in = tf.keras.Input(shape=ch1_X_train[0].shape)
    cn1_h1 = tf.keras.layers.Conv1D(32, 3, activation=None)(wave_in)
    bn1_h1 = tf.keras.layers.BatchNormalization()(cn1_h1)
    re1_h1 = tf.keras.layers.ReLU()(bn1_h1)
    mp1_h1 = tf.keras.layers.MaxPooling1D(pool_size=3,strides=1, padding='valid')(re1_h1)
    cn2_h2 = tf.keras.layers.Conv1D(64, 3, activation=None)(mp1_h1)
    bn2_h2 = tf.keras.layers.BatchNormalization()(cn2_h2)
    re2_h2 = tf.keras.layers.ReLU()(bn2_h2)
    mp2_h2 = tf.keras.layers.MaxPooling1D(pool_size=3,strides=1, padding='valid')(re2_h2)
    cn3_h3 = tf.keras.layers.Conv1D(128, 3, activation=None)(mp2_h2)
    bn3_h3 = tf.keras.layers.BatchNormalization()(cn3_h3)
    re3_h3 = tf.keras.layers.ReLU()(bn3_h3)

    mp3_h3 = tf.keras.layers.MaxPooling1D(pool_size=3,strides=1, padding='valid')(re3_h3)
    
    l1_h1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(mp3_h3)
    a1_h1 = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention')(l1_h1)
    f1_h1 = tf.keras.layers.Flatten()(a1_h1)
    ch1_pts = tf.keras.layers.Dense(len(X_train[0]),activation='sigmoid')(f1_h1)
    #ch2_pts = tf.keras.layers.Dense(len(X_train[0]),activation='sigmoid')(f1_h1)
    bp_out = tf.keras.layers.Dense(1)(f1_h1)
    
    
    model = keras.Model(
        inputs=[wave_in],
        outputs=[ch1_pts,bp_out],
    )
    opt = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(
        optimizer=opt,
        loss=[
            tf.keras.losses.BinaryCrossentropy(),
            'mse'
        ],
        loss_weights=[
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
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_dense_1_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
    callbacks_list = [checkpoint,history]

    model.fit(X_train, [ch1_X_dtw_train,sbp_train], validation_split=0.1, shuffle=True, epochs=50, batch_size=128, callbacks=callbacks_list)

    model.load_weights(weights_filepath)
    #model.save_weights(weights_filepath)

    preds = model.predict(X_test)

    np.save(save_directory + subject + '/X_test',X_test)
    np.save(save_directory + subject + '/BP_test',sbp_test)
    np.save(save_directory + subject + '/preds_0',preds[0])
    np.save(save_directory + subject + '/preds_1',preds[1])
    
    K.clear_session()
    
    print('Done!')
    
    