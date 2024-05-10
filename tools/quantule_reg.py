# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 20:48:19 2023

@author: Steve
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import *
from keras.models import *
from keras import backend as K
from keras.engine.topology import Layer
from keras.preprocessing import sequence
from keras.models import load_model
from scipy import stats

def quantile_loss(y_pred, y_true, r=0.5):
    greater_mask = K.cast((y_true <= y_pred), 'float32')
    smaller_mask = K.cast((y_true > y_pred), 'float32')
    return K.sum((r-1)*K.abs(smaller_mask*(y_true-y_pred)), -1)+K.sum(r*K.abs(greater_mask*(y_true-y_pred)), -1)


def tilted_loss(q, y_true, y_pred):
    e = (y_true-y_pred)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)


def lstm_1v(quantile=0.95,TIME_STEPS=3,INPUT_DIM=3):
    #Bi-LSTM
    input1 = Input(shape=(TIME_STEPS,1))
    lstm_out1 = LSTM(128,input_shape=(TIME_STEPS,1),return_sequences=True)(input1)
    #timeseries_drop1 = Dropout(0.2)(lstm_out1)
    #lstm_out2 = LSTM(256,input_shape=(TIME_STEPS,1),return_sequences=True)(lstm_out1)
    #timeseries_drop2 = Dropout(0.3)(lstm_out2)
    #lstm_out3 = LSTM(64,input_shape=(TIME_STEPS,1),return_sequences=True)(lstm_out2)
    #timeseries_drop3 = Dropout(0.2)(lstm_out3)
    attention_mul = attention_3d_block(lstm_out1)
    attention_flatten = Flatten()(attention_mul)
    #drop1 = Dropout(0.2)(attention_flatten)
    dense1 = Dense(units=64,activation="linear")(attention_flatten)
    dense2 = Dense(units=32,activation="linear")(dense1)
    
    dense3 = Dense(units=10,activation="linear")(dense2)

    dense4 = Dense(1)(dense3) #这里最后就应该是1，因为每一个输出结果向量长度都是1

    model = Model(inputs=[input1], outputs=dense4) 
    model.compile(loss=lambda y_t, y_p: tilted_loss(quantile, y_t, y_p), optimizer='adam') 
    return model







