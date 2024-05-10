# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 09:40:35 2023

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

def create_model(model_type,TIME_STEPS=3,INPUT_DIM=3,F_DIM=11):
    '''
    Parameters
    ----------
    model_type : str, '3v_6f','3v_4f','3v'
                'v' refers to vector, 'f' refers to factor
                and the number refers to the inputs the model considers
    ----------
    Returns
    A bi-lstm model with certain structure
    '''
    if model_type == '3v_6f':
        #Bi-LSTM
        input1 = Input(shape=(TIME_STEPS,INPUT_DIM))
        lstm_out1 = Bidirectional(LSTM(128, 
                       input_shape=(TIME_STEPS, INPUT_DIM), 
                       return_sequences=True),name='bilstm1')(input1)
        #timeseries_drop1 = Dropout(0.2)(lstm_out1)
        lstm_out2 = Bidirectional(LSTM(256, 
                       input_shape=(TIME_STEPS, INPUT_DIM), 
                       return_sequences=True),name='bilstm2')(lstm_out1)
        #timeseries_drop2 = Dropout(0.3)(lstm_out2)
        lstm_out3 = Bidirectional(LSTM(64, 
                       input_shape=(TIME_STEPS, INPUT_DIM), 
                       return_sequences=True),name='bilstm3')(lstm_out2)
        #timeseries_drop3 = Dropout(0.2)(lstm_out3)
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

        self_attention = Self_Attention(128)(fdense2)
        fdense3 = Dense(units=32,activation="sigmoid")(self_attention)
        fdense4 = Dense(units=8,activation="sigmoid")(fdense3)
        fdense_permute = Permute((2, 1), name='self_attention_permute')(fdense4)
        factor_output = GlobalAveragePooling1D()(fdense_permute)

        #合并
        merge = concatenate([dense2, factor_output])
        dense3 = Dense(units=10,activation="linear")(merge)

        dense4 = Dense(1)(dense3) #这里最后就应该是1，因为每一个输出结果向量长度都是1

        #因素端2
        input3 = Input(shape=(2))
        #dense5 = Dense(units=12,activation="linear")(input3)
        #dense6 = Dense(units=2,activation="linear")(dense5)

        #再次合并
        merge2 = concatenate([dense4, input3])
        dense7 = Dense(12,activation="linear")(merge2)
        output = Dense(1)(dense7)

        model = Model(inputs=[input1,input2,input3], outputs=output)
        
    elif model_type == '3v_4f':
        #Bi-LSTM
        input1 = Input(shape=(TIME_STEPS,INPUT_DIM))
        lstm_out1 = Bidirectional(LSTM(128, 
                       input_shape=(TIME_STEPS, INPUT_DIM), 
                       return_sequences=True),name='bilstm1')(input1)
        #timeseries_drop1 = Dropout(0.2)(lstm_out1)
        lstm_out2 = Bidirectional(LSTM(256, 
                       input_shape=(TIME_STEPS, INPUT_DIM), 
                       return_sequences=True),name='bilstm2')(lstm_out1)
        #timeseries_drop2 = Dropout(0.3)(lstm_out2)
        lstm_out3 = Bidirectional(LSTM(64, 
                       input_shape=(TIME_STEPS, INPUT_DIM), 
                       return_sequences=True),name='bilstm3')(lstm_out2)
        #timeseries_drop3 = Dropout(0.2)(lstm_out3)
        attention_mul = attention_3d_block(lstm_out3)
        attention_flatten = Flatten()(attention_mul)
        #drop1 = Dropout(0.2)(attention_flatten)
        dense1 = Dense(units=64,activation="linear")(attention_flatten)
        dense2 = Dense(units=32,activation="linear")(dense1)
        
        #因素端
        input2 = Input(shape=(4,1))
        fdense1 = Dense(units=32,activation="sigmoid")(input2)
        #drop2 = Dropout(0.2)(fdense1)
        fdense2 = Dense(units=128,activation="sigmoid")(fdense1)
        #drop3 = Dropout(0.1)(fdense2)

        self_attention = Self_Attention(128)(fdense2)
        fdense3 = Dense(units=32,activation="sigmoid")(self_attention)
        fdense4 = Dense(units=8,activation="sigmoid")(fdense3)
        fdense_permute = Permute((2, 1), name='self_attention_permute')(fdense4)
        factor_output = GlobalAveragePooling1D()(fdense_permute)

        #合并
        merge = concatenate([dense2, factor_output])
        dense3 = Dense(units=10,activation="linear")(merge)

        dense4 = Dense(1)(dense3) #这里最后就应该是1，因为每一个输出结果向量长度都是1


        model = Model(inputs=[input1,input2], outputs=dense4)
        
    elif model_type == '3v':
        #Bi-LSTM
        input1 = Input(shape=(TIME_STEPS,INPUT_DIM))
        lstm_out1 = Bidirectional(LSTM(128, 
                       input_shape=(TIME_STEPS, INPUT_DIM), 
                       return_sequences=True),name='bilstm1')(input1)
        #timeseries_drop1 = Dropout(0.2)(lstm_out1)
        lstm_out2 = Bidirectional(LSTM(256, 
                       input_shape=(TIME_STEPS, INPUT_DIM), 
                       return_sequences=True),name='bilstm2')(lstm_out1)
        #timeseries_drop2 = Dropout(0.3)(lstm_out2)
        lstm_out3 = Bidirectional(LSTM(64, 
                       input_shape=(TIME_STEPS, INPUT_DIM), 
                       return_sequences=True),name='bilstm3')(lstm_out2)
        #timeseries_drop3 = Dropout(0.2)(lstm_out3)
        attention_mul = attention_3d_block(lstm_out3)
        attention_flatten = Flatten()(attention_mul)
        #drop1 = Dropout(0.2)(attention_flatten)
        dense1 = Dense(units=64,activation="linear")(attention_flatten)
        dense2 = Dense(units=32,activation="linear")(dense1)
        
        dense3 = Dense(units=10,activation="linear")(dense2)

        dense4 = Dense(1)(dense3) #这里最后就应该是1，因为每一个输出结果向量长度都是1

        model = Model(inputs=[input1], outputs=dense4)

    elif model_type == '2v':
        #Bi-LSTM
        input1 = Input(shape=(TIME_STEPS,2))
        lstm_out1 = Bidirectional(LSTM(128, 
                       input_shape=(TIME_STEPS,2), 
                       return_sequences=True),name='bilstm1')(input1)
        #timeseries_drop1 = Dropout(0.2)(lstm_out1)
        lstm_out2 = Bidirectional(LSTM(256, 
                       input_shape=(TIME_STEPS,2), 
                       return_sequences=True),name='bilstm2')(lstm_out1)
        #timeseries_drop2 = Dropout(0.3)(lstm_out2)
        lstm_out3 = Bidirectional(LSTM(64, 
                       input_shape=(TIME_STEPS,2), 
                       return_sequences=True),name='bilstm3')(lstm_out2)
        #timeseries_drop3 = Dropout(0.2)(lstm_out3)
        attention_mul = attention_3d_block(lstm_out3)
        attention_flatten = Flatten()(attention_mul)
        #drop1 = Dropout(0.2)(attention_flatten)
        dense1 = Dense(units=64,activation="linear")(attention_flatten)
        dense2 = Dense(units=32,activation="linear")(dense1)
        
        dense3 = Dense(units=10,activation="linear")(dense2)

        dense4 = Dense(1)(dense3) #这里最后就应该是1，因为每一个输出结果向量长度都是1

        model = Model(inputs=[input1], outputs=dense4)    
    elif model_type == '1v':
        #Bi-LSTM
        input1 = Input(shape=(TIME_STEPS,1))
        lstm_out1 = Bidirectional(LSTM(128, 
                       input_shape=(TIME_STEPS,1), 
                       return_sequences=True),name='bilstm1')(input1)
        #timeseries_drop1 = Dropout(0.2)(lstm_out1)
        lstm_out2 = Bidirectional(LSTM(256, 
                       input_shape=(TIME_STEPS,1), 
                       return_sequences=True),name='bilstm2')(lstm_out1)
        #timeseries_drop2 = Dropout(0.3)(lstm_out2)
        lstm_out3 = Bidirectional(LSTM(64, 
                       input_shape=(TIME_STEPS,1), 
                       return_sequences=True),name='bilstm3')(lstm_out2)
        #timeseries_drop3 = Dropout(0.2)(lstm_out3)
        attention_mul = attention_3d_block(lstm_out3)
        attention_flatten = Flatten()(attention_mul)
        #drop1 = Dropout(0.2)(attention_flatten)
        dense1 = Dense(units=64,activation="linear")(attention_flatten)
        dense2 = Dense(units=32,activation="linear")(dense1)
        
        dense3 = Dense(units=10,activation="linear")(dense2)

        dense4 = Dense(1)(dense3) #这里最后就应该是1，因为每一个输出结果向量长度都是1

        model = Model(inputs=[input1], outputs=dense4)   
    elif model_type == 'lstm_1v':
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
    elif model_type == '1v_6f':
        #Bi-LSTM
        input1 = Input(shape=(TIME_STEPS,1))
        lstm_out1 = Bidirectional(LSTM(128, 
                       input_shape=(TIME_STEPS, 1), 
                       return_sequences=True),name='bilstm1')(input1)
        #timeseries_drop1 = Dropout(0.2)(lstm_out1)
        lstm_out2 = Bidirectional(LSTM(256, 
                       input_shape=(TIME_STEPS, 1), 
                       return_sequences=True),name='bilstm2')(lstm_out1)
        #timeseries_drop2 = Dropout(0.3)(lstm_out2)
        lstm_out3 = Bidirectional(LSTM(64, 
                       input_shape=(TIME_STEPS, 1), 
                       return_sequences=True),name='bilstm3')(lstm_out2)
        #timeseries_drop3 = Dropout(0.2)(lstm_out3)
        attention_mul = attention_3d_block(lstm_out3)
        attention_flatten = Flatten()(attention_mul)
        #drop1 = Dropout(0.2)(attention_flatten)
        dense1 = Dense(units=64,activation="linear")(attention_flatten)
        dense2 = Dense(units=32,activation="linear")(dense1)
        
        #因素端
        input2 = Input(shape=(4,1))
        fdense1 = Dense(units=32,activation="sigmoid")(input2)
        #drop2 = Dropout(0.2)(fdense1)
        fdense2 = Dense(units=128,activation="sigmoid")(fdense1)
        #drop3 = Dropout(0.1)(fdense2)

        self_attention = Self_Attention(128)(fdense2)
        fdense3 = Dense(units=32,activation="sigmoid")(self_attention)
        fdense4 = Dense(units=8,activation="sigmoid")(fdense3)
        fdense_permute = Permute((2, 1), name='self_attention_permute')(fdense4)
        factor_output = GlobalAveragePooling1D()(fdense_permute)

        #合并
        merge = concatenate([dense2, factor_output])
        dense3 = Dense(units=10,activation="linear")(merge)

        dense4 = Dense(1)(dense3) #这里最后就应该是1，因为每一个输出结果向量长度都是1

        #因素端2
        input3 = Input(shape=(2))
        #dense5 = Dense(units=12,activation="linear")(input3)
        #dense6 = Dense(units=2,activation="linear")(dense5)

        #再次合并
        merge2 = concatenate([dense4, input3])
        dense7 = Dense(12,activation="linear")(merge2)
        output = Dense(1)(dense7)

        model = Model(inputs=[input1,input2,input3], outputs=output)        
    model.compile(loss='mse', 
              optimizer='adam')     
    return model


class Meta_Model():
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

class Meta_Model_interprete():
    def __init__(self,LSTM_model,tree_model):
        '''
        Parameters
        ----------
        LSTM_model : well trained keras model
        meta_model : untrained tree-based model like xgboost
        '''
        self.lstm_model = LSTM_model
        self.tree_model = tree_model
        
    def fit(self,full_inputs,train_y,lr=1,v_num=6,v_dim=3,f_num=13):
        '''
        Parameters
        ----------
        full_inputs: np.array
            including input of vector_X and factor_X
            (vector_X: np.array
                input of LSTM model, (sample_num,time_step*vector_var_num)
            factor_X : np.array
                input of tree_model (sample_num,factor_var_num))
        train_y : np.array
            output, tabular data (sample_num,1)
        lr : float
            learning rate in the boosting step (0~1)
        '''
        #construcr meta training input
        vector_X = full_inputs[:,:v_num].reshape(full_inputs.shape[0],3,v_dim)
        lstm_pred = self.lstm_model.predict(vector_X).reshape(-1,1)
        
        factor_X = full_inputs[:,v_num:(v_num+f_num)]
        meta_X = np.concatenate((lstm_pred,factor_X),axis=1)
        
        #learn negative gradient of lstm_pred
        #we have: n_grad = -d(error)/d(lstm_pred) = -d(lstm_pred-train_y)**2/ d(lstm)
        n_grad = -(lstm_pred-train_y)
        self.tree_model.fit(meta_X,n_grad*lr)
        self.lr = lr
        self.v_num=v_num
        self.f_num=f_num
        self.v_dim=v_dim
    
    def predict(self,full_inputs):
        v_num=self.v_num
        f_num=self.f_num        
        vector_X = full_inputs[:,:v_num].reshape(full_inputs.shape[0],3,self.v_dim)
        factor_X = full_inputs[:,v_num:(v_num+f_num)]
        lstm_pred = self.lstm_model.predict(vector_X).reshape(-1,1)
        meta_X = np.concatenate((lstm_pred,factor_X),axis=1)
        meta_pred = self.tree_model.predict(meta_X).reshape(-1,1)*self.lr + lstm_pred
        
        return meta_pred


class Normalize_timeseris():
    def fit(self,data,norm_type=0):
        '''
        Parameters
        ----------
        data : 3d np.array
            be like (sample_num,time_step,variable_num=1)
        norm_type : TYPE, optional
            0 refers to minimax normalization
            1 refers to standard normalization. The default is 0.
            
        Returns
        -------
        return nothing. only used to find parameters for normalization
        '''
        self.norm_type = norm_type
        if self.norm_type==0:
            self.min_val = np.array([[np.min(data[:,1,:],axis=0)]])
            self.max_val = np.array([[np.max(data[:,1,:],axis=0)]])
        elif self.norm_type==1:
            self.std = np.array([[np.std(data[:,1,:],axis=0)]])
            self.mean = np.array([[np.mean(data[:,1,:],axis=0)]])
    
    def norm_transform(self,data):
        if self.norm_type==0:
            norm_data = (data-self.min_val)/(self.max_val-self.min_val)
        elif self.norm_type==1:
            norm_data = (data-self.meanl)/self.std
        return norm_data
    
    def unorm_transform(self,data):
        if self.norm_type==0:
            unorm_data = data*(self.max_val-self.min_val)+self.min_val
        elif self.norm_type==1:
            unorm_data = data*self.std+self.mean
        return unorm_data

class Normalize():
    def fit(self,data,norm_type=0):
        '''
        Parameters
        ----------
        data : 2d np.array
            be like (sample_num,variable_num)
        norm_type : TYPE, optional
            0 refers to minimax normalization
            1 refers to standard normalization. The default is 0.
            
        Returns
        -------
        return nothing. only used to find parameters for normalization
        '''
        self.norm_type = norm_type
        if self.norm_type==0:
            self.min_val = np.min(data,axis=0)
            self.max_val = np.max(data,axis=0)
        elif self.norm_type==1:
            self.std = np.std(data,axis=0)
            self.mean = np.mean(data,axis=0)
    
    def norm_transform(self,data):
        if self.norm_type==0:
            norm_data = (data-self.min_val)/(self.max_val-self.min_val)
        elif self.norm_type==1:
            norm_data = (data-self.meanl)/self.std
        return norm_data
    
    def unorm_transform(self,data):
        if self.norm_type==0:
            unorm_data = data*(self.max_val-self.min_val)+self.min_val
        elif self.norm_type==1:
            unorm_data = data*self.std+self.mean
        return unorm_data

def error_interval_cal(y_pred,y_true,a=0.9):
    '''
    Parameters
    ----------
    a : float,confidence coefficient.
    y_pred : 1d array, predicted y
    y_true : 1d array, true y

    Returns
    -------
    the error interval covers the errors in confidencial rate a.

    '''
    errors = y_pred.reshape(-1)-y_true.reshape(-1)
    error_interval = np.percentile(errors,[a-(1-a)/2,(1-a)/2])
    return error_interval


import numpy as np 
import pandas as pd
def coef_cal(X,y):
    '''
    Parameters
    ----------
    X : np.array (sample_num,var_num)
    y : np.array (sample_num)
    Returns
    -------
    coef_mat : dataframe, (var_num,coeftype_num)

    '''
    mat = np.concatenate((X,y.reshape(-1,1)),axis=1)
    mat = pd.DataFrame(mat)
    coef_list = ['pearson','spearman','kendall']
    coef_pearson = mat.corr('pearson').iloc[0:-1,-1]
    coef_spearman = mat.corr('spearman').iloc[0:-1,-1]
    coef_kendall = mat.corr('kendall').iloc[0:-1,-1]
    coef_mat = pd.DataFrame([coef_pearson,coef_spearman,coef_kendall],
                            index=coef_list)
    coef_mat = pd.DataFrame(coef_mat.values.T,columns=coef_list)
    
    return coef_mat


def onehot_encode(y,yo):
    y = y.reshape(-1)
    y = y.astype('int')
    yo = yo.astype('int')
    value_set = np.unique(yo)
    y_encode = np.zeros((len(y),len(value_set)))
    
    v_index = 0
    for value in value_set:
        for i in range(len(y)):
            if y[i]==value:
                y_encode[i,v_index]=1
        v_index+=1
    return y_encode.astype('int')

#metircs_cal
def R_cal(y_p,y_t):
    y_p = y_p.reshape(-1)
    y_t = y_t.reshape(-1)
    x1=0; x2=0; x3=0
    p_mean = np.mean(y_p)
    t_mean = np.mean(y_t)
    for i in range(len(y_p)):
        x1 += abs((y_p[i]-p_mean)*(y_t[i]-t_mean))
        x2 += (y_p[i]-p_mean)**2
        x3 += (y_t[i]-t_mean)**2
    r = x1/np.sqrt(x2*x3)
    return r
    
#计算MAE    
def mae_cal(y_p,y_t):
    y_p = y_p.reshape(-1)
    y_t = y_t.reshape(-1)
    x=0
    for i in range(len(y_p)):
        x += abs(y_p[i]-y_t[i])
    mae = x/len(y_p)
    return mae


#计算RMSE    
def rmse_cal(y_p,y_t):
    y_p = y_p.reshape(-1)
    y_t = y_t.reshape(-1)
    x=0
    for i in range(len(y_p)):
        x += (y_p[i]-y_t[i])**2
    rmse = np.sqrt(x/len(y_p))
    return rmse
#计算MSE    
def mse_cal(y_p,y_t):
    y_p = y_p.reshape(-1)
    y_t = y_t.reshape(-1)
    x=0
    for i in range(len(y_p)):
        x += (y_p[i]-y_t[i])**2
    mse = x/len(y_p)
    return mse

#计算MAPE   
def mape_cal(y_p,y_t):
    y_p = y_p.reshape(-1)
    y_t = y_t.reshape(-1)
    x=0
    for i in range(len(y_p)):
        x += abs(y_p[i]-y_t[i])/y_t[i]
    mape = x/len(y_p)*100
    return mape

def rsquared(x, y): 
    _, _, r_value, _, _ = stats.linregress(x.reshape(-1), y.reshape(-1)) 
    #a、b、r
    return r_value**2

def metrics_cal(y_p,y_t,axis=0):
    if axis==0:
        mat = np.array([mae_cal(y_p,y_t),rmse_cal(y_p,y_t),mape_cal(y_p,y_t),rsquared(y_p,y_t)]).reshape(1,-1)
        mat = pd.DataFrame(mat,columns=['mae','rmse','mape','R'])
    else:
        mat = np.array([mae_cal(y_p,y_t),rmse_cal(y_p,y_t),mape_cal(y_p,y_t),rsquared(y_p,y_t)]).reshape(-1,1)
        mat = pd.DataFrame(mat,index=['mae','rmse','mape','R'])
    return mat

def add_noise(y,intensity=0.1):
    org = np.random.randn(y.shape[0])
    norm = (org-np.min(org))/(np.max(org)-np.min(org))
    noise = abs(norm*intensity)
    noise = noise[np.argsort(noise)]
    y_noise = y+noise.reshape(-1,1)
    return y_noise

def add_noise2(y,intensity=0.7):
    # intensity = np.random.rand()*intensity
    noise_base = np.append(np.diff(y.reshape(-1)),0)
    org = np.random.rand(y.shape[0])
    norm = (org-np.min(org))/(np.max(org)-np.min(org))-0.5
    noise = norm*intensity*noise_base
    #noise = noise[np.argsort(noise)]
    y_noise = y+noise.reshape(-1,1)
    return y_noise

def add_discrete_noise(y,intensity=0.1,discrete=0.4):
    y = y.reshape(-1)
    noise = abs(np.random.randn(y.shape[0])*intensity)
    noise = noise[np.argsort(noise)]   
    dnoise = np.append([0],noise[1:]-noise[:-1])
    dis_list = np.random.choice([0,1],size=(y.shape[0]),p=[discrete,1-discrete])
    dis_dnoise = np.cumsum(dnoise*dis_list)+noise[0]
    
    return dis_dnoise+y
        

        
def find_singular(seq):
    seq = seq.reshape(-1)
    der2 = np.diff(np.diff(seq))     
    singular = np.concatenate(([0],der2<=np.min(der2),[0]))
    return singular.reshape(-1,1)
if __name__=='__main__':
    y = np.arange(10)
    yn = add_discrete_noise(y,intensity=2,discrete=0.4)
    seq = np.array([1,2,3,4,5,8,9,20,10])
    singular = find_singular(seq)


        
        
        
        
        
        
        
        
        
        