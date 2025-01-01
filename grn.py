import keras
import tensorflow as tf
import numpy as np 
from tqdm import tqdm
from tensorflow.keras import layers,optimizers
from sklearn.model_selection import train_test_split

class GatesResidualNetwork(layers.Layer):
    def __init__(self,units,dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout = dropout
        self.dense_1 = layers.Dense(units,activation=None)
        self.dense_2 = layers.Dense(units,activation=None)
        self.elu = layers.Activation('elu')
        self.gate_dense = layers.Dense(units,activation='sigmoid')
        self.add = layers.Add()
        self.dropout = layers.Dropout(dropout)
        self.layer_norm = layers.LayerNormalization()
    def call(self,inputs,context=None):
        x = self.dense_1(inputs)
        x = self.elu(x)
        if context is not None:
            context_transform = self.dense_2(context)
            context_transform = self.elu(context_transform)
            x = self.add([x,context_transform])
            #x = self.dropout(x)
            x = self.layer_norm(x)
            x = self.gate_dense(x)
            #x = self.add([inputs,x])
        return x 