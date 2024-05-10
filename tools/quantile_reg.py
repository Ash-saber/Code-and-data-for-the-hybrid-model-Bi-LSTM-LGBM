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

def attention_3d_block(inputs,TIME_STEPS=3):
    #input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

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

class Meta_Model_quantile():
    def __init__(self,LSTM_model,tree_model):
        '''
        Parameters
        ----------
        LSTM_model : well trained keras model
        meta_model : untrained tree-based model like xgboost
        '''
        self.lstm_model = LSTM_model
        self.tree_model = tree_model
        
    def fit(self,vector_X,factor_X,train_y,lr=1):
        '''
        Parameters
        ----------
        vector_X: np.array
            input of LSTM model, (sample_num,time_step,vector_var_num)
        factor_X : np.array
            input of tree_model (sample_num,factor_var_num)
        train_y : np.array
            output, tabular data (sample_num,1)
        lr : float
            learning rate in the boosting step (0~1)
        '''
        #construcr meta training input
        lstm_pred = self.lstm_model.predict(vector_X).reshape(-1,1)
        meta_X = np.concatenate((lstm_pred,factor_X),axis=1)
        
        #learn negative gradient of lstm_pred
        #we have: n_grad = -d(error)/d(lstm_pred) = -d(lstm_pred-train_y)**2/ d(lstm)
        n_grad = -(lstm_pred-train_y)
        self.tree_model.fit(meta_X,n_grad*lr)
        self.lr = lr
    
    def predict(self,vector_X,factor_X):
        lstm_pred = self.lstm_model.predict(vector_X).reshape(-1,1)
        meta_X = np.concatenate((lstm_pred,factor_X),axis=1)
        meta_pred = self.tree_model.predict(meta_X).reshape(-1,1)*self.lr + lstm_pred
        
        return meta_pred





