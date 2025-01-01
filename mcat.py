import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import add, concatenate, TimeDistributed, Concatenate
import numpy as np

import math, copy, time


class Encoder(layers.Layer):
    def __init__(self,layer, N):
        super(Encoder,self).__init__()
        self.layers = [] 
        for i in range(N):
            self.layers.append(layer)
        self.norm = LayerNorm(layer.size)
    
    def call(self,x,mask=None):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class LayerNorm(layers.Layer):
    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.a_2 = np.ones(features)
        self.b_2 = np.ones(features)
        self.eps = eps
    
    def call(self,x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True)
        out = self.a_2 * (x-mean)/(std + self.eps) + self.b_2
        return out

class SublayerConnection(layers.Layer):
    def __init__(self,size, dropout):
        super(SublayerConnection, self).__init__()
        #self.norm =layers.LayerNormalization() # LayerNorm(size)
        self.norm =LayerNorm(size)
        self.dropout = layers.Dropout(dropout)

    def call(self,x,sublayer):
        sub = sublayer(self.norm(x))
        return x + self.dropout(sub) 

class EncoderLayer(layers.Layer):
    def __init__(self,size,self_attt,feed_forward,dropout):
        super(EncoderLayer,self).__init__()
        self.self_attt = self_attt
        self.feed_forward = feed_forward
        self.sublayer_0 = SublayerConnection(size,dropout)
        self.sublayer_1 = SublayerConnection(size,dropout)
        self.size = size
    def call(self,x):
        lamb = lambda x: self.self_attt(x,x,x)
        x = self.sublayer_0(x,lamb)
        return self.sublayer_1(x,self.feed_forward)

def attention_with_pos(query, key, value, pos_k, pos_v, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class HAR_CNN(layers.Layer):
    def __init__(self,d_model,d_ff,filters,dropout=0.2):
        super(HAR_CNN,self).__init__()
        self.kernel_num = int(d_ff)
        self.filter_sizes = filters
        self.dropout = layers.Dropout(dropout)
        self.bn = layers.BatchNormalization(axis=1)#layers.LayerNormalization(axis=1)
        self.relu = layers.Activation('relu')
        self.encoders = []
        for i, filter_size in enumerate(self.filter_sizes):
            encoder = layers.Conv1D(filters=self.kernel_num, 
                                    kernel_size=filter_size,
                                    data_format='channels_first',
                                    padding = 'same'
                                   )
            self.encoders.append(encoder)

    def call(self, data):
        data = tf.cast(data, dtype=tf.float32)
        enc_outs = []
        for encoder in self.encoders:
            temp = tf.transpose(data, perm=[0, 2, 1])
            f_map = encoder(tf.transpose(data, perm=[0, 2, 1]))
            enc_ = f_map
            enc_ = self.relu(self.dropout(self.bn(enc_)))
            enc_outs.append(tf.expand_dims(enc_, axis=1))

        re = tf.divide(tf.reduce_sum(tf.concat(enc_outs, axis=1), axis=1),3)
        return tf.transpose(re, perm=[0, 2, 1])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = tf.cast(tf.shape(query)[-1], tf.float32)
    scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(d_k)

    if mask is not None:
        scores += (mask * -1e9)

    p_attn = tf.keras.activations.softmax(scores, axis=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return tf.matmul(p_attn, value), p_attn

class MultiHeadAttention(layers.Layer):
    def __init__(self,h,d_model,dropout=0.1):
        super(MultiHeadAttention,self).__init__()
        self.d_k = d_model // h
        self.h = h
        self.linears = [layers.Dense(d_model) for _ in range(4)]
        self.att = None
        self.dropout = layers.Dropout(dropout)

    def get_rel_pos(self, x):
        return max(self.k*-1, min(self.k, x))

    def call(self, query, key, value, mask=None):
        if mask is not None:
            mask = tf.expand_dims(mask, 1)
        nbatches = tf.shape(query)[0]

        query, key, value = \
            [tf.transpose(tf.reshape(l(x), (nbatches, -1, self.h, self.d_k)), perm=[0, 2, 1, 3])
             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask)

        x = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]), (nbatches, -1, self.h * self.d_k))
        return self.linears[-1](x)

class MCAT(layers.Layer):
    def __init__(self, hidden_dim, N, H, total_size, filters=[1, 3, 5]):
        super(MCAT, self).__init__()

        self.model = Encoder(
            EncoderLayer(hidden_dim, MultiHeadAttention(H, hidden_dim),
                         HAR_CNN(hidden_dim, hidden_dim, filters)
                         , 0.1),
            N
        )

    def call(self, x):
        return self.model(x)