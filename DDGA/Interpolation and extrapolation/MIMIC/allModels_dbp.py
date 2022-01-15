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

class Time2Vec(keras.layers.Layer):
    def __init__(self, kernel_size=1):
        super(Time2Vec, self).__init__(trainable=True, name='Time2VecLayer')
        self.k = kernel_size
    
    def build(self, input_shape):
        # trend
        self.wb = self.add_weight(name='wb',shape=(input_shape[1],),initializer='uniform',trainable=True)
        self.bb = self.add_weight(name='bb',shape=(input_shape[1],),initializer='uniform',trainable=True)
        # periodic
        self.wa = self.add_weight(name='wa',shape=(1, input_shape[1], self.k),initializer='uniform',trainable=True)
        self.ba = self.add_weight(name='ba',shape=(1, input_shape[1], self.k),initializer='uniform',trainable=True)
        super(Time2Vec, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        bias = self.wb * inputs + self.bb
        dp = K.dot(inputs, self.wa) + self.ba
        wgts = K.sin(dp) # or K.cos(.)

        ret = K.concatenate([K.expand_dims(bias, -1), wgts], -1)
        ret = K.reshape(ret, (-1, inputs.shape[1]*(self.k+1)))
        return ret
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]*(self.k + 1))
    
from tensorflow_addons.layers import MultiHeadAttention

class AttentionBlock(keras.Model):
    def __init__(self, name='AttentionBlock', num_heads=2, head_size=128, ff_dim=None, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)

        if ff_dim is None:
            ff_dim = head_size

        self.attention = MultiHeadAttention(num_heads=num_heads, head_size=head_size, dropout=dropout)
        self.attention_dropout = keras.layers.Dropout(dropout)
        self.attention_norm = keras.layers.LayerNormalization(epsilon=1e-6)

        self.ff_conv1 = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')
        # self.ff_conv2 at build()
        self.ff_dropout = keras.layers.Dropout(dropout)
        self.ff_norm = keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.ff_conv2 = keras.layers.Conv1D(filters=input_shape[-1], kernel_size=1) 

    def call(self, inputs):
        x = self.attention([inputs, inputs])
        x = self.attention_dropout(x)
        x = self.attention_norm(inputs + x)

        x = self.ff_conv1(x)
        x = self.ff_conv2(x)
        x = self.ff_dropout(x)

        x = self.ff_norm(inputs + x)
        return x
    
class ModelTrunk(keras.Model):
    def __init__(self, name='ModelTrunk', time2vec_dim=1, num_heads=2, head_size=128, ff_dim=None, num_layers=1, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.time2vec = Time2Vec(kernel_size=time2vec_dim)
        if ff_dim is None:
            ff_dim = head_size
        self.dropout = dropout
        self.attention_layers = [AttentionBlock(num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)]

        
    def call(self, inputs):
        time_embedding = keras.layers.TimeDistributed(self.time2vec)(inputs)
        x = K.concatenate([inputs, time_embedding], -1)
        for attention_layer in self.attention_layers:
            x = attention_layer(x)

        return K.reshape(x, (-1, x.shape[1] * x.shape[2])) # flat vector of features out
    
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
    
    exp = 'e591'
     
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
    #bp_data = np.concatenate((np.reshape(dbp_data,(len(dbp_data),1)),np.reshape(sbp_data,(len(sbp_data),1))),axis=1)
    
    # training_size = 0.8
    # x_train = X_data[0:int(len(X_data)*training_size)]     
    # x_dtw_train = X_dtw_data[0:int(len(X_dtw_data)*training_size)]
    # sbp_train = sbp_data[0:int(len(sbp_data)*training_size)]
    # dbp_train = dbp_data[0:int(len(dbp_data)*training_size)]
    #bp_train = bp_data[0:int(len(bp_data)*training_size)]
    
    # x_test = X_data[int(len(X_data)*training_size):]     
    # x_dtw_test = X_dtw_data[int(len(X_dtw_data)*training_size):]
    # sbp_test = sbp_data[int(len(sbp_data)*training_size):]
    # dbp_test = dbp_data[int(len(dbp_data)*training_size):]
    #bp_test = bp_data[int(len(bp_data)*training_size):]

    BP_test = dbp_data
    
    max_bp = np.amax(BP_test)
    min_bp = np.amin(BP_test)
    bucket_num = 1
    buckets = defaultdict(dict)
    bucket_num = 1
    bucket_size = 10
    bucketBottom = min_bp
    while True:
        if ((bucketBottom + bucket_size) - max_bp) > 3:
            break
        buckets['bucket' + str(bucket_num)] = np.where((bucketBottom <= BP_test)&(BP_test < (bucketBottom + bucket_size)))[0]
        bucket_num += 1
        bucketBottom += bucket_size
        
    for testBucket in buckets.keys():
        train_indices,test_indices = [],[]
        for bucket in buckets.keys():
            if bucket == testBucket:
                test_indices = buckets[testBucket].tolist()
            else:
                train_indices += buckets[bucket].tolist()
    
        if len(train_indices) < 1:
            continue
        elif len(test_indices) < 1:
            continue
    
        x_train = X_data[np.asarray(train_indices)]     
        x_dtw_train = X_dtw_data[np.asarray(train_indices)]
        dbp_train = dbp_data[np.asarray(train_indices)]

        x_test = X_data[np.asarray(test_indices)]     
        x_dtw_test = X_dtw_data[np.asarray(test_indices)]
        dbp_test = dbp_data[np.asarray(test_indices)]

        X_train = x_train
        X_test = x_test
        X_dtw_train = x_dtw_train
        X_dtw_test = x_dtw_test

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        # Channel 1 inputs
        ch1_X_train = X_train
        ch1_X_test = X_test
        ch1_X_dtw_train = x_dtw_train
        ch1_X_dtw_test = x_dtw_test
        
        X_train = np.reshape(X_train,(len(X_train),len(X_train[0]),1))
        X_test = np.reshape(X_test,(len(X_test),len(X_test[0]),1))

        ch1_X_train = np.reshape(ch1_X_train,(len(ch1_X_train),len(ch1_X_train[0]),1))
        ch1_X_test = np.reshape(ch1_X_test,(len(ch1_X_test),len(ch1_X_test[0]),1))

        ##################################
        # BiLSTM+Attn
        
        save_directory = "Results/bilstm_attn/personalized/dbp/" + exp + "/"    
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        if not os.path.exists(save_directory + subject + '/' + testBucket + '/models/'):
            os.makedirs(save_directory + subject + '/' + testBucket + '/models/')
            
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
        if not os.path.exists(save_directory + subject + '/' + testBucket + '/models/'):
            os.makedirs(save_directory + subject + '/' + testBucket + '/models/')
        weights_filepath = save_directory + subject + '/' + testBucket + '/models/best_weights_exp1.hdf5'
        checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
        callbacks_list = [checkpoint,history]

        model.fit(X_train, dbp_train, validation_split=0.1, shuffle=True, epochs=50, batch_size=128, callbacks=callbacks_list)

        model.load_weights(weights_filepath)
        #model.save_weights(weights_filepath)

        preds = model.predict(X_test)

        np.save(save_directory + subject + '/' + testBucket + '/X_test',X_test)
        np.save(save_directory + subject + '/' + testBucket + '/BP_test',dbp_test)
        np.save(save_directory + subject + '/' + testBucket + '/preds',preds)

        K.clear_session()

        save_directory = "Results/GA_bilstm_attn/personalized/dbp/" + exp + "/"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            
        if not os.path.exists(save_directory + subject + '/' + testBucket + '/models/'):
            os.makedirs(save_directory + subject + '/' + testBucket + '/models/')
        
        wave_in = tf.keras.Input(shape=X_train[0].shape)
        l1_h1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(wave_in)
        a1_h1 = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                           kernel_regularizer=keras.regularizers.l2(1e-4),
                           bias_regularizer=keras.regularizers.l1(1e-4),
                           attention_regularizer_weight=1e-4,
                           name='Attention')(l1_h1)
        f1_h1 = tf.keras.layers.Flatten()(a1_h1)
        ch1_pts = tf.keras.layers.Dense(len(X_train[0]),activation='sigmoid')(f1_h1)
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
                0.5,
                1.0
            ]
        )

        model.summary()

        history = History()
        # checkpoint
        if not os.path.exists(save_directory + subject + '/' + testBucket + '/models/'):
            os.makedirs(save_directory + subject + '/' + testBucket + '/models/')
        weights_filepath = save_directory + subject + '/' + testBucket + '/models/best_weights_exp1.hdf5'
        checkpoint = ModelCheckpoint(weights_filepath, monitor='val_dense_1_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
        callbacks_list = [checkpoint,history]

        model.fit(X_train, [ch1_X_dtw_train,dbp_train], validation_split=0.1, shuffle=True, epochs=50, batch_size=128, callbacks=callbacks_list)

        model.load_weights(weights_filepath)
        #model.save_weights(weights_filepath)

        preds = model.predict(X_test)

        np.save(save_directory + subject + '/' + testBucket + '/X_test',X_test)
        np.save(save_directory + subject + '/' + testBucket + '/BP_test',dbp_test)
        np.save(save_directory + subject + '/' + testBucket + '/preds_0',preds[0])
        np.save(save_directory + subject + '/' + testBucket + '/preds_1',preds[1])

        K.clear_session()

        
        
        ##################################
        # CNN+BiLSTM+Attn
        
        save_directory = "Results/cnn_bilstm_attn/personalized/dbp/" + exp + "/"    
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        if not os.path.exists(save_directory + subject + '/' + testBucket + '/models/'):
            os.makedirs(save_directory + subject + '/' + testBucket + '/models/')
            
        print('Init model...')


        wave_in = tf.keras.Input(shape=X_train[0].shape)
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
        if not os.path.exists(save_directory + subject + '/' + testBucket + '/models/'):
            os.makedirs(save_directory + subject + '/' + testBucket + '/models/')
        weights_filepath = save_directory + subject + '/' + testBucket + '/models/best_weights_exp1.hdf5'
        checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
        callbacks_list = [checkpoint,history]

        model.fit(X_train, dbp_train, validation_split=0.1, shuffle=True, epochs=50, batch_size=128, callbacks=callbacks_list)

        model.load_weights(weights_filepath)
        #model.save_weights(weights_filepath)

        preds = model.predict(X_test)

        np.save(save_directory + subject + '/' + testBucket + '/X_test',X_test)
        np.save(save_directory + subject + '/' + testBucket + '/BP_test',dbp_test)
        np.save(save_directory + subject + '/' + testBucket + '/preds',preds)

        K.clear_session()

        save_directory = "Results/GA_cnn_bilstm_attn/personalized/dbp/" + exp + "/"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            
        if not os.path.exists(save_directory + subject + '/' + testBucket + '/models/'):
            os.makedirs(save_directory + subject + '/' + testBucket + '/models/')
        
        wave_in = tf.keras.Input(shape=X_train[0].shape)
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
                0.5,
                1.0
            ]
        )

        model.summary()

        history = History()
        # checkpoint
        if not os.path.exists(save_directory + subject + '/' + testBucket + '/models/'):
            os.makedirs(save_directory + subject + '/' + testBucket + '/models/')
        weights_filepath = save_directory + subject + '/' + testBucket + '/models/best_weights_exp1.hdf5'
        checkpoint = ModelCheckpoint(weights_filepath, monitor='val_dense_1_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
        callbacks_list = [checkpoint,history]

        model.fit(X_train, [ch1_X_dtw_train,dbp_train], validation_split=0.1, shuffle=True, epochs=50, batch_size=128, callbacks=callbacks_list)

        model.load_weights(weights_filepath)
        #model.save_weights(weights_filepath)

        preds = model.predict(X_test)

        np.save(save_directory + subject + '/' + testBucket + '/X_test',X_test)
        np.save(save_directory + subject + '/' + testBucket + '/BP_test',dbp_test)
        np.save(save_directory + subject + '/' + testBucket + '/preds_0',preds[0])
        np.save(save_directory + subject + '/' + testBucket + '/preds_1',preds[1])

        K.clear_session()
        
        ##################################
        # Transformer
        
        save_directory = "Results/transformer/personalized/dbp/" + exp + "/"    
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        if not os.path.exists(save_directory + subject + '/' + testBucket + '/models/'):
            os.makedirs(save_directory + subject + '/' + testBucket + '/models/')
            
        print('Init model...')


        wave_in = tf.keras.Input(shape=X_train[0].shape)
        tr1_h1 = ModelTrunk(num_heads=4, head_size=64, ff_dim=None, num_layers=1)(wave_in)
        f1_h1 = tf.keras.layers.Flatten()(tr1_h1)
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
        if not os.path.exists(save_directory + subject + '/' + testBucket + '/models/'):
            os.makedirs(save_directory + subject + '/' + testBucket + '/models/')
        weights_filepath = save_directory + subject + '/' + testBucket + '/models/best_weights_exp1.hdf5'
        checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
        callbacks_list = [checkpoint,history]

        model.fit(X_train, dbp_train, validation_split=0.1, shuffle=True, epochs=50, batch_size=128, callbacks=callbacks_list)

        model.load_weights(weights_filepath)
        #model.save_weights(weights_filepath)

        preds = model.predict(X_test)

        np.save(save_directory + subject + '/' + testBucket + '/X_test',X_test)
        np.save(save_directory + subject + '/' + testBucket + '/BP_test',dbp_test)
        np.save(save_directory + subject + '/' + testBucket + '/preds',preds)

        K.clear_session()

        save_directory = "Results/GA_transformer/personalized/dbp/" + exp + "/"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            
        if not os.path.exists(save_directory + subject + '/' + testBucket + '/models/'):
            os.makedirs(save_directory + subject + '/' + testBucket + '/models/')
        
        wave_in = tf.keras.Input(shape=X_train[0].shape)
        tr1_h1 = ModelTrunk(num_heads=4, head_size=64, ff_dim=None, num_layers=1)(wave_in)
        f1_h1 = tf.keras.layers.Flatten()(tr1_h1)
        ch1_pts = tf.keras.layers.Dense(len(X_train[0]),activation='sigmoid')(f1_h1)
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
                0.5,
                1.0
            ]
        )

        model.summary()

        history = History()
        # checkpoint
        if not os.path.exists(save_directory + subject + '/' + testBucket + '/models/'):
            os.makedirs(save_directory + subject + '/' + testBucket + '/models/')
        weights_filepath = save_directory + subject + '/' + testBucket + '/models/best_weights_exp1.hdf5'
        checkpoint = ModelCheckpoint(weights_filepath, monitor='val_dense_1_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
        callbacks_list = [checkpoint,history]

        model.fit(X_train, [ch1_X_dtw_train,dbp_train], validation_split=0.1, shuffle=True, epochs=50, batch_size=128, callbacks=callbacks_list)

        model.load_weights(weights_filepath)
        #model.save_weights(weights_filepath)

        preds = model.predict(X_test)

        np.save(save_directory + subject + '/' + testBucket + '/X_test',X_test)
        np.save(save_directory + subject + '/' + testBucket + '/BP_test',dbp_test)
        np.save(save_directory + subject + '/' + testBucket + '/preds_0',preds[0])
        np.save(save_directory + subject + '/' + testBucket + '/preds_1',preds[1])

        K.clear_session()

