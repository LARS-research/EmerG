import random
import torch
import numpy as np
import pandas as pd
import pickle
import os
from torch.utils.data import Dataset, DataLoader


class TaobaoADbaseDataset(Dataset):
    """
    Load a base Taobao Dataset 
    """
    def __init__(self, dataset_name, df, description, device):
        super(TaobaoADbaseDataset, self).__init__()
        self.dataset_name = dataset_name
        self.df = df
        self.length = len(df)
        self.name2array = {name: torch.from_numpy(np.array(list(df[name])).reshape([self.length, -1])).to(device) \
                                        for name in df.columns}
        self.format(description)
        self.features = [name for name in df.columns if name != 'clk']
        self.label = 'clk'

        

    def format(self, description):
        for name, size, type in description:
            if type == 'spr' or type == 'seq':
                self.name2array[name] = self.name2array[name].to(torch.long)
            elif type == 'ctn':
                self.name2array[name] = self.name2array[name].to(torch.float32)
            elif type == 'label':
                pass
            else:
                raise ValueError('unkwon type {}'.format(type))
                
    def __getitem__(self, index):
        return {name: self.name2array[name][index] for name in self.features}, \
                self.name2array[self.label][index].squeeze()

    def __len__(self):
        return self.length
    
class TaobaoADbaseDataset_aldi(Dataset):
    """
    Load a base Taobao Dataset 
    """
    def __init__(self, dataset_name, df, description, device):
        super(TaobaoADbaseDataset_aldi, self).__init__()
        self.dataset_name = dataset_name
        self.df = df
        self.length = len(df)
        self.name2array = {name: torch.from_numpy(np.array(list(df[name])).reshape([self.length, -1])).to(device) \
                                        for name in df.columns}
        self.format(description)
        self.features = [name for name in df.columns if name != 'clk']
        self.label = 'clk'

        

    def format(self, description):
        for name, size, type in description:
            if type == 'spr' or type == 'seq':
                self.name2array[name] = self.name2array[name].to(torch.long)
            elif type == 'ctn':
                self.name2array[name] = self.name2array[name].to(torch.float32)
            elif type == 'label':
                pass
            else:
                raise ValueError('unkwon type {}'.format(type))
        #'cate_id', 'customer', 'campaign_id'
        self.name2array['item_id_neg'] = self.name2array['item_id_neg'].to(torch.long)
        self.name2array['cate_id_neg'] = self.name2array['cate_id_neg'].to(torch.long)
        self.name2array['customer_neg'] = self.name2array['customer_neg'].to(torch.long)
        self.name2array['campaign_id_neg'] = self.name2array['campaign_id_neg'].to(torch.long)
        self.name2array['price_neg'] = self.name2array['price_neg'].to(torch.long)
        self.name2array['brand_neg'] = self.name2array['brand_neg'].to(torch.long)
        self.name2array['pid_neg'] = self.name2array['pid_neg'].to(torch.long)
                
                
    def __getitem__(self, index):
        return {name: self.name2array[name][index] for name in self.features}

    def __len__(self):
        return self.length

class TaobaoADmetaDataLoader(Dataset):
    def __init__(self, dataset_name, df, description, device):
        super(TaobaoADmetaDataLoader, self).__init__()
        self.dataset_name = dataset_name
        self.df = df
        self.length = len(df.groupby('item_id'))
        self.name2array = {name: [torch.from_numpy(np.array(list(df_group[name])).reshape([len(df_group), -1])).to(device) for item_id, df_group in df.groupby('item_id')]\
                                    for name in df.columns}
        self.format(description)
        self.features = [name for name in df.columns if name != 'clk']
        self.label = 'clk'

    def format(self, description):
        for name, size, type in description:
            if type == 'spr' or type == 'seq':
                self.name2array[name] = [self.name2array[name][i].to(torch.long) for i in range(self.length)]
            elif type == 'ctn':
                self.name2array[name] = [self.name2array[name][i].to(torch.float32) for i in range(self.length)]
            elif type == 'label':
                pass
            else:
                raise ValueError('unkwon type {}'.format(type))

    def __getitem__(self, index):
        return {name: self.name2array[name][index] for name in self.features}, \
                self.name2array[self.label][index].squeeze()

    def __len__(self):
        return self.length


class TaobaoADColdStartDataLoader(object):
    """
    Load all splitted Taobao Dataset for cold start setting

    :param dataset_path: Taobao dataset path
    """

    def __init__(self, dataset_name, dataset_path, device, bsz=32, shuffle=True):
        assert os.path.exists(dataset_path), '{} does not exist'.format(dataset_path)
        with open(dataset_path, 'rb+') as f:
            data = pickle.load(f)
        self.dataset_name = dataset_name
        self.dataloaders = {}
        self.description = data['description']
        for key, df in data.items():
            if key == 'description':
                continue
            if "train_base" in key:
                self.dataloaders[key] = DataLoader(TaobaoADbaseDataset(dataset_name, df, self.description, device), batch_size=bsz, shuffle=shuffle)
            else:
                self.dataloaders[key] = DataLoader(TaobaoADbaseDataset(dataset_name, df, self.description, device), batch_size=bsz, shuffle=shuffle)
            if key == 'metaE_a':
                self.dataloaders['metaa_item2group'] = DataLoader(TaobaoADmetaDataLoader(dataset_name, df, self.description, device), batch_size=1, shuffle=False)
            elif key == 'metaE_b':
                self.dataloaders['metab_item2group'] = DataLoader(TaobaoADmetaDataLoader(dataset_name, df, self.description, device), batch_size=1, shuffle=False)
            elif key == 'metaE_c':
                self.dataloaders['metac_item2group'] = DataLoader(TaobaoADmetaDataLoader(dataset_name, df, self.description, device), batch_size=1, shuffle=False)
            elif key == 'metaE_d':
                self.dataloaders['metad_item2group'] = DataLoader(TaobaoADmetaDataLoader(dataset_name, df, self.description, device), batch_size=1, shuffle=False)
            elif key == 'train_warm_a':
                self.dataloaders['warma_item2group'] = DataLoader(TaobaoADmetaDataLoader(dataset_name, df, self.description, device), batch_size=1, shuffle=False)
            elif key == 'train_warm_b':
                self.dataloaders['warmb_item2group'] = DataLoader(TaobaoADmetaDataLoader(dataset_name, df, self.description, device), batch_size=1, shuffle=False)
            elif key == 'train_warm_c':
                self.dataloaders['warmc_item2group'] = DataLoader(TaobaoADmetaDataLoader(dataset_name, df, self.description, device), batch_size=1, shuffle=False)
            elif key == 'test':
                self.dataloaders['test_item2group'] = DataLoader(TaobaoADmetaDataLoader(dataset_name, df, self.description, device), batch_size=1, shuffle=False)
                      
                
        
        self.keys = list(self.dataloaders.keys())
        self.item_features = ['item_id', 'cate_id', 'customer', 'campaign_id', 'price', 'brand', 'pid']

    def __getitem__(self, name):
        assert name in self.keys, '{} not in keys of datasets'.format(name)
        return self.dataloaders[name]
    



