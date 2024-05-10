# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 21:56:31 2023

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
from keras.layers import Reshape
# first way attention
def attention_3d_block(inputs,TIME_STEPS=3):
    #input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

#self-attention
class Self_Attention(Layer):
 
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        #inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3,input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
 
        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它
 
    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
 
        #print("WQ.shape",WQ.shape)
 
        #print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)
 
        QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))
 
        QK = QK / (64**0.5)
 
        QK = K.softmax(QK)
 
        #print("QK.shape",QK.shape)
 
        V = K.batch_dot(QK,WV)
 
        return V
 
    def compute_output_shape(self, input_shape):
 
        return (input_shape[0],input_shape[1],self.output_dim)
def create_model2(model_type,TIME_STEPS=3,INPUT_DIM=3,F_DIM=11,lstmc_num=3,
                  lstms1_num=1,lstms2_num=1,lstms3_num=1,mlpc_num=3,
                  mlps1_num=1, mlps2_num=1, mlps3_num=1):
    '''
    Parameters
    ----------
    model_type : str, '2v_4f','2v',
                'v' refers to vector, 'f' refers to factor
                and the number refers to the inputs the model considers
    ----------
    Returns
    A bi-lstm model with certain structure
    '''
    
    lstms_range = [64,128,256]
    mlps_range = [16,32,64]
    if model_type == '2v_4f':
        #Bi-LSTM
        input1 = Input(shape=(TIME_STEPS,INPUT_DIM))
        lstm_out1 = Bidirectional(LSTM(lstms_range[lstms1_num], 
                       input_shape=(TIME_STEPS, INPUT_DIM), 
                       return_sequences=True),name='bilstm1')(input1)
        #timeseries_drop1 = Dropout(0.2)(lstm_out1)
        lstm_out2 = Bidirectional(LSTM(lstms_range[lstms2_num], 
                       input_shape=(TIME_STEPS, INPUT_DIM), 
                       return_sequences=True),name='bilstm2')(lstm_out1)
        #timeseries_drop2 = Dropout(0.3)(lstm_out2)
        lstm_out3 = Bidirectional(LSTM(lstms_range[lstms3_num], 
                       input_shape=(TIME_STEPS, INPUT_DIM), 
                       return_sequences=True),name='bilstm3')(lstm_out2)
        #timeseries_drop3 = Dropout(0.2)(lstm_out3)
        if lstmc_num==0:
            attention_mul = attention_3d_block(lstm_out1)
        elif lstmc_num==1:
            attention_mul = attention_3d_block(lstm_out2)
        elif lstmc_num==2:
            attention_mul = attention_3d_block(lstm_out3)

        attention_flatten = Flatten()(attention_mul)
        #drop1 = Dropout(0.2)(attention_flatten)
        dense1 = Dense(units=64,activation="linear")(attention_flatten)
        dense2 = Dense(units=32,activation="linear")(dense1)
        
        #因素端
        input2 = Input(shape=(F_DIM,1))
        fdense1 = Dense(units=32,activation="sigmoid")(input2)
        #drop2 = Dropout(0.2)(fdense1)
        fdense2 = Dense(units=128,activation="sigmoid")(fdense1)
        #drop3 = Dropout(0.1)(fdense2)

        #self_attention = Self_Attention(128)(fdense2)
        #fdense3 = Dense(units=32,activation="sigmoid")(self_attention)
        fdense4 = Dense(units=8,activation="sigmoid")(fdense2)
        #fdense_permute = Permute((2, 1), name='self_attention_permute')(fdense4)
        factor_output = GlobalAveragePooling1D()(fdense4)

        #合并
        merge = concatenate([dense2, factor_output])
        dense3 = Dense(units=10,activation="linear")(merge)

        dense4 = Dense(1)(dense3) #这里最后就应该是1，因为每一个输出结果向量长度都是1

        dense7 = Dense(12,activation="linear")(dense4)
        output = Dense(1)(dense7)

        model = Model(inputs=[input1,input2], outputs=output)
        

    elif model_type == '2v':
        #Bi-LSTM
        input1 = Input(shape=(TIME_STEPS,INPUT_DIM))
        lstm_out1 = Bidirectional(LSTM(lstms_range[lstms1_num], 
                       input_shape=(TIME_STEPS, INPUT_DIM), 
                       return_sequences=True),name='bilstm1')(input1)
        #timeseries_drop1 = Dropout(0.2)(lstm_out1)
        lstm_out2 = Bidirectional(LSTM(lstms_range[lstms2_num], 
                       input_shape=(TIME_STEPS, INPUT_DIM), 
                       return_sequences=True),name='bilstm2')(lstm_out1)
        #timeseries_drop2 = Dropout(0.3)(lstm_out2)
        lstm_out3 = Bidirectional(LSTM(lstms_range[lstms3_num], 
                       input_shape=(TIME_STEPS, INPUT_DIM), 
                       return_sequences=True),name='bilstm3')(lstm_out2)
        #timeseries_drop3 = Dropout(0.2)(lstm_out3)
        if lstmc_num==0:
            attention_mul = attention_3d_block(lstm_out1)
        elif lstmc_num==1:
            attention_mul = attention_3d_block(lstm_out2)
        elif lstmc_num==2:
            attention_mul = attention_3d_block(lstm_out3)
        attention_flatten = Flatten()(attention_mul)
        #drop1 = Dropout(0.2)(attention_flatten)
        dense1 = Dense(units=64,activation="linear")(attention_flatten)
        dense2 = Dense(units=32,activation="linear")(dense1)
        
        dense3 = Dense(units=10,activation="linear")(dense2)

        dense4 = Dense(1)(dense3) #这里最后就应该是1，因为每一个输出结果向量长度都是1

        model = Model(inputs=[input1], outputs=dense4)

    elif model_type == '2v_interpret':
        #Bi-LSTM
        input1 = Input(shape=(TIME_STEPS*2))
        input_reshape = Reshape(input1.shape[0],3,2)(input1)
        lstm_out1 = Bidirectional(LSTM(lstms_range[lstms1_num], 
                       input_shape=(TIME_STEPS, INPUT_DIM), 
                       return_sequences=True),name='bilstm1')(input_reshape)
        #timeseries_drop1 = Dropout(0.2)(lstm_out1)
        lstm_out2 = Bidirectional(LSTM(lstms_range[lstms2_num], 
                       input_shape=(TIME_STEPS, INPUT_DIM), 
                       return_sequences=True),name='bilstm2')(lstm_out1)
        #timeseries_drop2 = Dropout(0.3)(lstm_out2)
        lstm_out3 = Bidirectional(LSTM(lstms_range[lstms3_num], 
                       input_shape=(TIME_STEPS, INPUT_DIM), 
                       return_sequences=True),name='bilstm3')(lstm_out2)
        #timeseries_drop3 = Dropout(0.2)(lstm_out3)
        if lstmc_num==0:
            attention_mul = attention_3d_block(lstm_out1)
        elif lstmc_num==1:
            attention_mul = attention_3d_block(lstm_out2)
        elif lstmc_num==2:
            attention_mul = attention_3d_block(lstm_out3)
        attention_flatten = Flatten()(attention_mul)
        #drop1 = Dropout(0.2)(attention_flatten)
        dense1 = Dense(units=64,activation="linear")(attention_flatten)
        dense2 = Dense(units=32,activation="linear")(dense1)
        
        dense3 = Dense(units=10,activation="linear")(dense2)

        dense4 = Dense(1)(dense3) #这里最后就应该是1，因为每一个输出结果向量长度都是1

        model = Model(inputs=[input1], outputs=dense4)

    elif model_type == 'mlp':
        input1 = Input(shape=(F_DIM+TIME_STEPS*INPUT_DIM))
        if mlpc_num==0:
            dense3 = Dense(units=mlps_range[mlps1_num],activation="linear")(input1)
        elif mlpc_num==1:
            dense1 = Dense(units=mlps_range[mlps1_num],activation="linear")(input1)
            dense3 = Dense(units=mlps_range[mlps2_num],activation="relu")(dense1)
        elif mlpc_num==2:
            dense1 = Dense(units=mlps_range[mlps1_num],activation="linear")(input1)
            dense2 = Dense(units=mlps_range[mlps2_num],activation="relu")(dense1)
            dense3 = Dense(units=mlps_range[mlps3_num],activation="relu")(dense2)
        dense4 = Dense(1)(dense3)
        model = Model(inputs=[input1], outputs=dense4)   
        
    model.compile(loss='mse', 
              optimizer='adam')     
    return model