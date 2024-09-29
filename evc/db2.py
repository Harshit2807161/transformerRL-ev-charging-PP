#import datetime
#now = datetime.datetime.now
from .common import pj
import torch as tt
import torch.nn as nn
from .rnn import *



import numpy as np
class PricePredictor:
    def __init__(self, path, past, future, model, auto_load=False):
        self.past, self.future = past, future
        self.model = model
        self.path = path
        if auto_load: self.load()
    
    def load(self): self.model.load_state_dict(tt.load(self.path, map_location=tt.device('cuda')))
    def save(self): tt.save(self.model.state_dict(),self.path)

    @tt.no_grad()
    def __call__(self, X):
        X = tt.tensor(X, dtype=tt.float32)
        if self.future<2: 
            return self.model(X.reshape(1,self.past,1))[0]
        else:
            x = tt.clone(X).reshape(1,self.past,1)
            y = self.model(x)[0,0]
            preds = [y.item()]
            for _ in range(self.future-1):
                x = x.numpy()
                x[:, 0:-1, :] = x[:, 1:, :]
                x[:, -1, :] = y
                x=tt.tensor(x, dtype=tt.float32)
                y =self.model(x)[0,0]
                preds.append(y.item())
            return np.array(preds)  
    
    @staticmethod
    def get_oracle_24_0(auto_load = False):
        return __class__(path=None, past = 24, future = 0, model=None)
    
    @staticmethod
    def get_oracle_24_1(auto_load = False):
        return __class__(path=None, past = 24, future = 1, model=None)
    
    @staticmethod
    def get_oracle_24_24(auto_load = False):
        return __class__(path=None, past = 24, future = 24, model=None)
    
    #======================================================
    # MODELS
    #======================================================

    default_i2h_sizes=[50, 50, 50, 50]
    default_i2o_sizes=[50, 50, 50, 50]
    default_o2o_sizes=[50, 50, 50, 50]
    default_fc_layers=[100,]

    default_i2o_activation=tt.sigmoid
    default_o2o_activation=tt.sigmoid

    default_fc_act = (nn.ELU, {})

    @staticmethod
    def lstm_uni_24_1(auto_load = False):
        return __class__(
            path =      pj('__models__/pp/lstm_uni_24_1.h5' ), 
            past =      24,
            future =    1, 
            auto_load = auto_load,
            model    = XRNN(
                coreF=LSTM,
                bidir=False,
                fc_layers=__class__.default_fc_layers,
                fc_act=__class__.default_fc_act,
                input_size=1,      
                i2h_sizes=__class__.default_i2h_sizes,
                i2o_sizes=__class__.default_i2o_sizes,
                o2o_sizes=__class__.default_o2o_sizes,
                i2o_activation=__class__.default_i2o_activation,
                o2o_activation=__class__.default_o2o_activation,
                dropout=0.0,        
                batch_first=True,
                hypers=None,
                dtype=tt.float32,)
            )

    @staticmethod
    def lstmA_uni_24_1(auto_load = False):
        return __class__(
            path =      pj('__models__/pp/lstmA_uni_24_1.h5' ), 
            past =      24,
            future =    1, 
            auto_load = auto_load,
            model    = XARNN(
                coreF=LSTM,
                bidir=False,
                fc_layers=__class__.default_fc_layers,
                fc_act=__class__.default_fc_act,
                input_size=1,      
                i2h_sizes=__class__.default_i2h_sizes,
                i2o_sizes=__class__.default_i2o_sizes,
                o2o_sizes=__class__.default_o2o_sizes,
                i2o_activation=__class__.default_i2o_activation,
                o2o_activation=__class__.default_o2o_activation,
                dropout=0.0,        
                batch_first=True,
                hypers=None,
                dtype=tt.float32,)
            )

    @staticmethod
    def lstm_bi_24_1(auto_load = False):
        return __class__(
            path =      pj('__models__/pp/lstm_bi_24_1.h5' ), 
            past =      24,
            future =    1, 
            auto_load = auto_load,
            model    = XRNN(
                coreF=LSTM,
                bidir=True,
                fc_layers=__class__.default_fc_layers,
                fc_act=__class__.default_fc_act,
                input_size=1,      
                i2h_sizes=__class__.default_i2h_sizes,
                i2o_sizes=__class__.default_i2o_sizes,
                o2o_sizes=__class__.default_o2o_sizes,
                i2o_activation=__class__.default_i2o_activation,
                o2o_activation=__class__.default_o2o_activation,
                dropout=0.0,        
                batch_first=True,
                hypers=None,
                dtype=tt.float32,)
            )

    @staticmethod
    def lstmA_bi_24_1(auto_load = False):
        return __class__(
            path =      pj('__models__/pp/lstmA_bi_24_1.h5' ), 
            past =      24,
            future =    1, 
            auto_load = auto_load,
            model    = XARNN(
                coreF=LSTM,
                bidir=True,
                fc_layers=__class__.default_fc_layers,
                fc_act=__class__.default_fc_act,
                input_size=1,      
                i2h_sizes=__class__.default_i2h_sizes,
                i2o_sizes=__class__.default_i2o_sizes,
                o2o_sizes=__class__.default_o2o_sizes,
                i2o_activation=__class__.default_i2o_activation,
                o2o_activation=__class__.default_o2o_activation,
                dropout=0.0,        
                batch_first=True,
                hypers=None,
                dtype=tt.float32,)
            )
    
    @staticmethod
    def janet_uni_24_1(auto_load = False):
        return __class__(
            path =      pj('__models__/pp/janet_uni_24_1.h5' ), 
            past =      24,
            future =    1, 
            auto_load = auto_load,
            model    = XRNN(
                coreF=JANET,
                bidir=False,
                fc_layers=__class__.default_fc_layers,
                fc_act=__class__.default_fc_act,
                input_size=1,      
                i2h_sizes=__class__.default_i2h_sizes,
                i2o_sizes=__class__.default_i2o_sizes,
                o2o_sizes=__class__.default_o2o_sizes,
                i2o_activation=__class__.default_i2o_activation,
                o2o_activation=__class__.default_o2o_activation,
                dropout=0.0,        
                batch_first=True,
                hypers=None,
                dtype=tt.float32,)
            )

    @staticmethod
    def janetA_uni_24_1(auto_load = False):
        return __class__(
            path =      pj('__models__/pp/janetA_uni_24_1.h5' ), 
            past =      24,
            future =    1, 
            auto_load = auto_load,
            model    = XARNN(
                coreF=JANET,
                bidir=False,
                fc_layers=__class__.default_fc_layers,
                fc_act=__class__.default_fc_act,
                input_size=1,      
                i2h_sizes=__class__.default_i2h_sizes,
                i2o_sizes=__class__.default_i2o_sizes,
                o2o_sizes=__class__.default_o2o_sizes,
                i2o_activation=__class__.default_i2o_activation,
                o2o_activation=__class__.default_o2o_activation,
                dropout=0.0,        
                batch_first=True,
                hypers=None,
                dtype=tt.float32,)
            )

    @staticmethod
    def janet_bi_24_1(auto_load = False):
        return __class__(
            path =      pj('__models__/pp/janet_bi_24_1.h5' ), 
            past =      24,
            future =    1, 
            auto_load = auto_load,
            model    = XRNN(
                coreF=JANET,
                bidir=True,
                fc_layers=__class__.default_fc_layers,
                fc_act=__class__.default_fc_act,
                input_size=1,      
                i2h_sizes=__class__.default_i2h_sizes,
                i2o_sizes=__class__.default_i2o_sizes,
                o2o_sizes=__class__.default_o2o_sizes,
                i2o_activation=__class__.default_i2o_activation,
                o2o_activation=__class__.default_o2o_activation,
                dropout=0.0,        
                batch_first=True,
                hypers=None,
                dtype=tt.float32,)
            )

    @staticmethod
    def janetA_bi_24_1(auto_load = False):
        return __class__(
            path =      pj('__models__/pp/janetA_bi_24_1.h5' ), 
            past =      24,
            future =    1, 
            auto_load = auto_load,
            model    = XARNN(
                coreF=JANET,
                bidir=True,
                fc_layers=__class__.default_fc_layers,
                fc_act=__class__.default_fc_act,
                input_size=1,      
                i2h_sizes=__class__.default_i2h_sizes,
                i2o_sizes=__class__.default_i2o_sizes,
                o2o_sizes=__class__.default_o2o_sizes,
                i2o_activation=__class__.default_i2o_activation,
                o2o_activation=__class__.default_o2o_activation,
                dropout=0.0,        
                batch_first=True,
                hypers=None,
                dtype=tt.float32,)
            )
    
    @staticmethod
    def mgru_uni_24_1(auto_load = False):
        return __class__(
            path =      pj('__models__/pp/mgru_uni_24_1.h5' ), 
            past =      24,
            future =    1, 
            auto_load = auto_load,
            model    = XRNN(
                coreF=MGRU,
                bidir=False,
                fc_layers=__class__.default_fc_layers,
                fc_act=__class__.default_fc_act,
                input_size=1,      
                i2h_sizes=__class__.default_i2h_sizes,
                i2o_sizes=__class__.default_i2o_sizes,
                o2o_sizes=__class__.default_o2o_sizes,
                i2o_activation=__class__.default_i2o_activation,
                o2o_activation=__class__.default_o2o_activation,
                dropout=0.0,        
                batch_first=True,
                hypers=None,
                dtype=tt.float32,)
            )

    @staticmethod
    def mgruA_uni_24_1(auto_load = False):
        return __class__(
            path =      pj('__models__/pp/mgruA_uni_24_1.h5' ), 
            past =      24,
            future =    1, 
            auto_load = auto_load,
            model    = XARNN(
                coreF=MGRU,
                bidir=False,
                fc_layers=__class__.default_fc_layers,
                fc_act=__class__.default_fc_act,
                input_size=1,      
                i2h_sizes=__class__.default_i2h_sizes,
                i2o_sizes=__class__.default_i2o_sizes,
                o2o_sizes=__class__.default_o2o_sizes,
                i2o_activation=__class__.default_i2o_activation,
                o2o_activation=__class__.default_o2o_activation,
                dropout=0.0,        
                batch_first=True,
                hypers=None,
                dtype=tt.float32,)
            )

    @staticmethod
    def mgru_bi_24_1(auto_load = False):
        return __class__(
            path =      pj('__models__/pp/mgru_bi_24_1.h5' ), 
            past =      24,
            future =    1, 
            auto_load = auto_load,
            model    = XRNN(
                coreF=MGRU,
                bidir=True,
                fc_layers=__class__.default_fc_layers,
                fc_act=__class__.default_fc_act,
                input_size=1,      
                i2h_sizes=__class__.default_i2h_sizes,
                i2o_sizes=__class__.default_i2o_sizes,
                o2o_sizes=__class__.default_o2o_sizes,
                i2o_activation=__class__.default_i2o_activation,
                o2o_activation=__class__.default_o2o_activation,
                dropout=0.0,        
                batch_first=True,
                hypers=None,
                dtype=tt.float32,)
            )

    @staticmethod
    def mgruA_bi_24_1(auto_load = False):
        return __class__(
            path =      pj('__models__/pp/mgruA_bi_24_1.h5' ), 
            past =      24,
            future =    1, 
            auto_load = auto_load,
            model    = XARNN(
                coreF=MGRU,
                bidir=True,
                fc_layers=__class__.default_fc_layers,
                fc_act=__class__.default_fc_act,
                input_size=1,      
                i2h_sizes=__class__.default_i2h_sizes,
                i2o_sizes=__class__.default_i2o_sizes,
                o2o_sizes=__class__.default_o2o_sizes,
                i2o_activation=__class__.default_i2o_activation,
                o2o_activation=__class__.default_o2o_activation,
                dropout=0.0,        
                batch_first=True,
                hypers=None,
                dtype=tt.float32,)
            )
    
# SAMPLE ARGS

"""
lstm_uni_24_1,lstmA_uni_24_1,lstm_bi_24_1,lstmA_bi_24_1,janet_uni_24_1,janetA_uni_24_1,janet_bi_24_1,janetA_bi_24_1,mgru_uni_24_1,mgruA_uni_24_1,mgru_bi_24_1,mgruA_bi_24_1
"""
# model    = XARNN(
#                 coreF=MGRU,
#                 bidir=False,
#                 fc_layers=__class__.default_fc_layers,
#                 fc_act=(nn.ReLU, {}),
#                 fc_last_act=None,
#                 fc_output=True,
#                 fc_bias=True,
#                 input_size=1,      
#                 i2h_sizes=__class__.default_i2h_sizes,      
#                 i2o_sizes=None,  
#                 o2o_sizes=None,  
#                 dropout=0.0,        
#                 batch_first=True,
#                 i2h_bias = True, 
#                 i2o_bias = True,
#                 o2o_bias = True,
#                 i2h_activations=None,
#                 i2o_activation=None,
#                 o2o_activation=None,
#                 last_activation=None,
#                 hypers=None,
#                 return_sequences=False,
#                 stack_output=False, 
#                 dtype=tt.float32,
#                 device=None,)







