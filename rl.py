#  [markdown]
import argparse
args=None
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #now_ts = basic.stamp_now()
    parser.add_argument('--data_dir', type=str, default='data', help='Base directory Containing Data')
    parser.add_argument('--test_set', type=str, default='test', help='File name for testing data') 
    parser.add_argument('--train_set', type=str, default='train', help='File name for training data') #
    parser.add_argument('--val_set', type=str, default='val', help='File name for validation data')

    parser.add_argument('--seq_len', type=int, default=24, help='Sequence length - no of past time steps')

    parser.add_argument('--result_dir', type=str, default='result',  
                        help='Base Directory for saving visualizations') #
    parser.add_argument('--rla_names', type=str, default='ddpg_01' , help='csv model names') #
    # default='lstm_uni_24_1,lstmA_uni_24_1,janet_uni_24_1,janetA_uni_24_1'
    # default='get_oracle_24_1'

    parser.add_argument('--epochs', type=int, default=10_00_000, help='epochs') #
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size') #
    parser.add_argument('--nval', type=int, default=10, 
                help='how many epochs to validate after, during training, should be less or equal to epochs')

    parser.add_argument('--lr', type=float, default=0.0004, help='lr') #
    parser.add_argument('--lref', type=float, default=2.5, help='lr end factor')

    #For PPO disc
    '''
    parser.add_argument('--lr', type=float, default=0.00004, help='lr') #
    parser.add_argument('--lref', type=float, default=.25, help='lr end factor')
    '''
    #parser.add_argument('--es_patience', type=float, default=10, help='early stop patience') #
    #parser.add_argument('--es_delta', type=float, default=0.0001, help='early stop delta') #
    parser.add_argument('--buff', type=int, default=1, help='if True(1), uses i2o and o2o layers')
    parser.add_argument('--verbose', type=int, default=2, help='verbose')
    parser.add_argument('--do_train', type=int, default=1,  #
        help='if 0, skips training and evaluates only. If 1 then initialize new model, if 2 then loads existing model')
    parser.add_argument('--do_vis', type=int, default=1, help='if true, visulaizes predictions')
    parser.add_argument('--nfig', type=int, default=0, help='no of figures for manual testing')
    '''
    parser.add_argument('--n_steps', type=int, default=256, help='HYP')
    parser.add_argument('--n_epochs', type=int, default=30, help='HYP')
    parser.add_argument('--clip_range', type=float, default=0.2, help='HYP')
    '''
    args = parser.parse_args()
else:
    raise Exception(f'Should not be imported!')

# python rl.py --model_names=lstm_uni_24_1,lstmA_uni_24_1 --rla_names=dqn_01 --epochs=1000
# python rl.py --model_names=janet_uni_24_1,janetA_uni_24_1

# 
#import tensorflow as tf
#tf.config.set_visible_devices([], 'GPU')

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
# #import pandas as pd
import torch as tt
# import torch.nn as nn
# import torch.optim as oo

from evc.common import now, pj
from evc.drl import RLA
from evc.common import create_default_dirs
from evc.db2_transformers import PricePredictor
from evc.price import PriceData, SeqDataset
create_default_dirs()

START_TIMESTAMP = now()
#  [markdown]
# # Data

# 

if args.test_set:
    csv_test = pj(f'{args.data_dir}/{args.test_set}.csv')
    assert os.path.exists(csv_test), f'test-set not found @ {csv_test}'
else:
    csv_test=None

if args.val_set:
    csv_val = pj(f'{args.data_dir}/{args.val_set}.csv')
    assert os.path.exists(csv_val), f'val-set not found @ {csv_val}'
else:
    csv_val=None

if args.do_train and args.train_set:
    csv_train = pj(f'{args.data_dir}/{args.train_set}.csv')
    assert os.path.exists(csv_train), f'train-set not found @ {csv_train}'
else:
    csv_train=None


average_past = 24
seqlen = args.seq_len
cols = ('PRICE',)
input_size = len(cols)
do_normalize=False
device = tt.device("cpu")

if csv_test is None:
    ds_test = None
else:
    ds_test  = PriceData(csv=csv_test, reverse=False)

if csv_val is None:
    ds_val=None
else:
    ds_val =  PriceData(csv=csv_val, reverse=False)
    
if args.do_train and (csv_train is not None):
    ds_train =  PriceData(csv=csv_train, reverse=False)
else:
    ds_train=None

#  [markdown]
# # Choose price predictor
model_names_train = [
    #"autoformer_normal_train.pkl",
    #"autoformer_basic_train.pkl",
    #"informer_basic_train.pkl",
    "patchtst_basic_train.pkl"
]

model_names_test = [
    #"autoformer_normal_test.pkl",
    #"autoformer_basic_test.pkl",
    #"informer_basic_test.pkl",
    "patchtst_basic_test.pkl"

]

model_names_val = [
    #"autoformer_normal_val.pkl",
    "autoformer_basic_val.pkl",
    "informer_basic_val.pkl",
    "patchtst_basic_val.pkl"
]

for i in range(len(model_names_train)):
    model_train = model_names_train[i]
    model_val = model_names_val[i]
    model_test = model_names_test[i]
    past = 24
    future = 1
    path = f"/home/user/DRL-shivendu/EVC_rl-final/EVC_rl-final/model_predictions/model_predictions/{model_train}"
    net_train = PricePredictor(path, past, future)

    path = f"/home/user/DRL-shivendu/EVC_rl-final/EVC_rl-final/model_predictions/model_predictions/{model_test}"
    net_test = PricePredictor(path, past, future)

    path = f"/home/user/DRL-shivendu/EVC_rl-final/EVC_rl-final/model_predictions/model_predictions/{model_val}"
    net_val = PricePredictor(path, past, future)

    trainer_call = getattr(RLA, f'{args.rla_names}')
    trainer_call(
            ds_train, ds_val, ds_test, average_past, 
            args.lr, args.lref, args.batch_size, args.epochs, args.nval, args.result_dir, args.nfig, args.do_vis,
            net_train,net_test,net_val,model_train)
    print("##############################################################################################################################################################################")
    print("##############################################################################################################################################################################")
    print(f"#############################################################[{model_names_train[i]} has finished execution]#################################################################")
    print("##############################################################################################################################################################################")
    print("##############################################################################################################################################################################")
