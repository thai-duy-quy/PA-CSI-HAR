import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import add, concatenate, TimeDistributed, Concatenate

import numpy as np
import math 

from mcat import MCAT
from position_encoding import GRE,PE
from transformer_encoder import Transfomer


class Two_Stream_Model(layers.Layer):
    def __init__(self,hlayers,vlayers,hheads,vheads,K,sample,num_class, maxlen):
        super(Two_Stream_Model,self).__init__()
        self.transfomer = MCAT(270,hlayers,hheads,500)
        self.kernel_num = 128
        self.kernel_num_v = 16
        self.filter_sizes = [20,40]
        self.filter_sizes_v = [2,4]
        self.sample = sample
        self.pos_endcoding = GRE(270,500,K)
        self.pos_endcoding_v = GRE(2000,30,K)
        self.maxlen = maxlen

        self.relu = layers.Activation('relu')

        if vlayers == 0:
            self.v_transformer = None
            self.dense =  layers.Dense(num_class, input_dim=270)
        else: 
            self.v_transformer = Transfomer(2000,vlayers,vheads)
            self.dense =  layers.Dense(num_class, input_dim=self.kernel_num * len(self.filter_sizes) + self.kernel_num_v * len(self.filter_sizes_v))
        
        self.dense2 = layers.Dense(num_class,input_dim = self.kernel_num * len(self.filter_sizes))
        self.dropout_rate = 0.5
        self.dropout = layers.Dropout(self.dropout_rate)
        self.encoders = []
        self.encoders_v = []

        for i, filter_size in enumerate(self.filter_sizes):
            encoder = layers.Conv1D(filters=self.kernel_num, 
                                    kernel_size=filter_size,
                                    data_format='channels_first'
                                   )
            self.encoders.append(encoder)
        for i, filter_size in enumerate(self.filter_sizes_v):
            encoder_v = layers.Conv1D(filters=self.kernel_num_v, 
                                    kernel_size=filter_size,
                                    data_format='channels_first'
                                   )
            self.encoders_v.append(encoder_v)
    
    def _aggregate(self,o,v=None):
        enc_outs = []
        enc_outs_v = []
        for encoder in self.encoders:
            f_map = encoder(tf.transpose(o, perm=[0, 2, 1]))
            enc_ = self.relu(f_map)
            k_h = enc_.shape[2]
            enc_ = layers.MaxPooling1D(data_format='channels_first',pool_size=k_h)(enc_)
            enc_ = tf.squeeze(enc_, axis=-1)
            enc_outs.append(enc_)
        encoding = self.dropout(tf.concat(enc_outs, axis=1))
        q_re = self.relu(encoding)
        if self.v_transformer is not None:
            for encoder in self.encoders_v:
                f_map = encoder(tf.transpose(v, perm=[0, 2, 1]))
                enc_ = self.relu(f_map)
                k_h = enc_.shape[2]
                enc_ = layers.MaxPooling1D(data_format='channels_first',pool_size=k_h)(enc_)
                enc_ = tf.squeeze(enc_, axis=-1)
                enc_outs_v.append(enc_)
            encoding_v = self.dropout(tf.concat(enc_outs_v, axis=1))
            v_re = self.relu(encoding_v)
            q_re = tf.concat([q_re,v_re], axis=-1)
        return q_re

    def call(self,data):
        data = tf.cast(data, dtype=tf.float32)
        d1 = tf.shape(data)[0]
        d2 = tf.shape(data)[1]
        d3 = tf.shape(data)[2]
        x = tf.reshape(data, (d1, d2//self.sample, self.sample, d3))
        x = tf.reduce_sum(x,axis=-2)
        x = tf.divide(x,self.sample)
        x = self.pos_endcoding(x)
        x = self.transfomer(x)

        if self.v_transformer is not None:
            y = tf.reshape(data,(-1,1000,9,30))
            y = tf.reduce_sum(y,axis=-2)
            y = tf.transpose(y,perm=[0, 2, 1])
            y = self.v_transformer(y)
            re = self._aggregate(x,y)
        else:
            re = self._aggregate(x)
        return re






