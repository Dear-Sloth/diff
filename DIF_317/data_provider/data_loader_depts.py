

import os
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.data_utils import group_values
from datetime import datetime
from utils.timefeatures import time_features
from sklearn.preprocessing import StandardScaler
import pickle as pkl

import warnings
warnings.filterwarnings('ignore')



class Dataset_Caiso(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path='caiso_20130101_20210630.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.args = args
    
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.dim_datetime_feats = np.shape(self.data_stamp)[-1]

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        from statsmodels.tsa.stattools import adfuller
        """
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(np.shape(data))
        print(self.args.dataset_name)
        result = adfuller(data[:,-1])
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        """
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['week'] = df_stamp.date.apply(lambda row: row.week, 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
    
        print("> {}-shape".format(self.flag), np.shape(self.data_x), np.shape(self.data_y))

        self.data_stamp = data_stamp

    def __getitem__(self, index):

        # ========================================================================
        # if (self.set_type==0) and (self.args.dataset_name in ["ECL", "traffic"]) and (self.args.model in ["DDPM"]):
        #     index = np.random.randint(len(self.data_x) - self.seq_len - self.pred_len + 1)

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]


        # =================================================================================
        # global time steps
        idx = 0
        timestamps1 = np.asarray(list(range(s_begin, s_end)))
        timestamps2 = np.asarray(list(range(r_begin, r_end)))
        #print(timestamps1.shape,timestamps2.shape)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, idx, timestamps1, timestamps2, len(self.data_x)
        
    def __len__(self):

        return len(self.data_x) - self.seq_len - self.pred_len + 1
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_Production(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path='production.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.args = args
    
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.dim_datetime_feats = np.shape(self.data_stamp)[-1]

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        from statsmodels.tsa.stattools import adfuller
        """
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(np.shape(data))
        print(self.args.dataset_name)
        result = adfuller(data[:,-1])
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        """
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['week'] = df_stamp.date.apply(lambda row: row.week, 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
    
        print("> {}-shape".format(self.flag), np.shape(self.data_x), np.shape(self.data_y))

        self.data_stamp = data_stamp

    def __getitem__(self, index):

        # ========================================================================
        # if (self.set_type==0) and (self.args.dataset_name in ["ECL", "traffic"]) and (self.args.model in ["DDPM"]):
        #     index = np.random.randint(len(self.data_x) - self.seq_len - self.pred_len + 1)

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]


        # =================================================================================
        # global time steps
        idx = 0
        timestamps1 = np.asarray(list(range(s_begin, s_end)))
        timestamps2 = np.asarray(list(range(r_begin, r_end)))
        #print(timestamps1.shape,timestamps2.shape)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, idx, timestamps1, timestamps2, len(self.data_x)
        
    def __len__(self):

        return len(self.data_x) - self.seq_len - self.pred_len + 1
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    
class Dataset_Solar(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path='solar_energy.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.args = args
    
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.dim_datetime_feats = np.shape(self.data_stamp)[-1]

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        from statsmodels.tsa.stattools import adfuller
        """
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(np.shape(data))
        print(self.args.dataset_name)
        result = adfuller(data[:,-1])
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        """
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['week'] = df_stamp.date.apply(lambda row: row.week, 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
    
        print("> {}-shape".format(self.flag), np.shape(self.data_x), np.shape(self.data_y))

        self.data_stamp = data_stamp

    def __getitem__(self, index):

        # ========================================================================
        # if (self.set_type==0) and (self.args.dataset_name in ["ECL", "traffic"]) and (self.args.model in ["DDPM"]):
        #     index = np.random.randint(len(self.data_x) - self.seq_len - self.pred_len + 1)

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]


        # =================================================================================
        # global time steps
        idx = 0
        timestamps1 = np.asarray(list(range(s_begin, s_end)))
        timestamps2 = np.asarray(list(range(r_begin, r_end)))
        #print(timestamps1.shape,timestamps2.shape)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, idx, timestamps1, timestamps2, len(self.data_x)
        
    def __len__(self):

        return len(self.data_x) - self.seq_len - self.pred_len + 1
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
