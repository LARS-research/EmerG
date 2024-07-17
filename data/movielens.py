import random
import numpy as np
import pandas as pd
import pickle
import os
import torch
from torch.utils.data import Dataset, DataLoader


class Movielens1MbaseDataset(Dataset):
    """
    Load a base Movielens Dataset 
    """
    def __init__(self, dataset_name, df, description, device, maml_episode=False):
        super(Movielens1MbaseDataset, self).__init__()
        self.dataset_name = dataset_name
        self.df = df
        self.length = len(df)
        self.name2array = {name: torch.from_numpy(np.array(list(df[name])).reshape([self.length, -1])).to(device) \
                                        for name in df.columns}
        self.format(description)
        self.features = [name for name in df.columns if name != 'rating']
        self.label = 'rating'

        self.maml_episode = maml_episode
        if maml_episode:
            user_count = self.df.groupby(['user_id']).size().reset_index(name='user_count')
            self.final_index = user_count[(user_count.user_count > 13) & (user_count.user_count < 100)].user_id.unique()

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
        if self.maml_episode:
            user_id = self.final_index[index]
            index = self.df[self.df.user_id == user_id].index.to_list()
            random.shuffle(index)
        return {name: self.name2array[name][index] for name in self.features}, \
                self.name2array[self.label][index].squeeze()
    


    def __len__(self):
        if self.maml_episode:
            return len(self.final_index)
        return self.length
    

class Movielens1MbaseDataset_aldi(Dataset):
    """
    Load a base Movielens Dataset 
    """
    def __init__(self, dataset_name, df, description, device, maml_episode=False):
        super(Movielens1MbaseDataset_aldi, self).__init__()
        self.dataset_name = dataset_name
        self.df = df
        self.length = len(df)
        self.name2array = {name: torch.from_numpy(np.array(list(df[name])).reshape([self.length, -1])).to(device) \
                                        for name in df.columns}
        self.format(description)
        self.features = [name for name in df.columns if name != 'rating']
        self.label = 'rating'

        self.maml_episode = maml_episode
        if maml_episode:
            user_count = self.df.groupby(['user_id']).size().reset_index(name='user_count')
            self.final_index = user_count[(user_count.user_count > 13) & (user_count.user_count < 100)].user_id.unique()

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
        self.name2array['year_neg'] = self.name2array['year_neg'].to(torch.long)
        self.name2array['title_neg'] = self.name2array['title_neg'].to(torch.long)
        self.name2array['genres_neg'] = self.name2array['genres_neg'].to(torch.long)
                
    def __getitem__(self, index):
        if self.maml_episode:
            user_id = self.final_index[index]
            index = self.df[self.df.user_id == user_id].index.to_list()
            random.shuffle(index)
        return {name: self.name2array[name][index] for name in self.features}
    
    def __len__(self):
        return self.length

class MovieLens1MmetaDataLoader(Dataset):
    def __init__(self, dataset_name, df, description, device, maml_episode=False):
        super(MovieLens1MmetaDataLoader, self).__init__()
        self.dataset_name = dataset_name
        self.df = df
        self.length = len(df.groupby('item_id'))
        #self.name2array = {name: torch.from_numpy(np.array([(np.array(list(df_group[name])).reshape([len(df_group), -1])).tolist() for item_id, df_group in df.groupby('item_id')])).to(device)\
                                    #for name in df.columns}
        self.name2array = {name: [torch.from_numpy(np.array(list(df_group[name])).reshape([len(df_group), -1])).to(device) for item_id, df_group in df.groupby('item_id')]\
                                    for name in df.columns}
        self.format(description)
        self.features = [name for name in df.columns if name != 'rating']
        self.label = 'rating'

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


class MovieLens1MColdStartDataLoader(object):
    """
    Load all splitted MovieLens 1M Dataset for cold start setting

    :param dataset_path: MovieLens dataset path
    """

    def __init__(self, dataset_name, dataset_path, device, bsz=32, shuffle=True, maml_episode=False):
        assert os.path.exists(dataset_path), '{} does not exist'.format(dataset_path)
        with open(dataset_path, 'rb+') as f:
            data = pickle.load(f)
        self.dataset_name = dataset_name
        self.dataloaders = {}
        self.description = data['description']
        #print(self.description.keys())
        for key, df in data.items():
            if key == 'description':
                continue
            if "train_base" in key:
                self.dataloaders[key] = DataLoader(Movielens1MbaseDataset(dataset_name, df, self.description, device, maml_episode=False), batch_size=bsz, shuffle=False)
            elif 'metaE' not in key:
                self.dataloaders[key] = DataLoader(Movielens1MbaseDataset(dataset_name, df, self.description, device, maml_episode=maml_episode), batch_size=bsz if not maml_episode else 1, shuffle=shuffle)
                
            else:
                self.dataloaders[key] = DataLoader(Movielens1MbaseDataset(dataset_name, df, self.description, device, maml_episode=False), batch_size=bsz, shuffle=False)
            if key == 'metaE_a':
                self.dataloaders['metaa_item2group'] = DataLoader(MovieLens1MmetaDataLoader(dataset_name, df, self.description, device, maml_episode=False), batch_size=1, shuffle=False)
            elif key == 'metaE_b':
                self.dataloaders['metab_item2group'] = DataLoader(MovieLens1MmetaDataLoader(dataset_name, df, self.description, device, maml_episode=False), batch_size=1, shuffle=False)
            elif key == 'metaE_c':
                self.dataloaders['metac_item2group'] = DataLoader(MovieLens1MmetaDataLoader(dataset_name, df, self.description, device, maml_episode=False), batch_size=1, shuffle=False)
            elif key == 'metaE_d':
                self.dataloaders['metad_item2group'] = DataLoader(MovieLens1MmetaDataLoader(dataset_name, df, self.description, device, maml_episode=False), batch_size=1, shuffle=False)
            elif key == 'train_warm_a':
                self.dataloaders['warma_item2group'] = DataLoader(MovieLens1MmetaDataLoader(dataset_name, df, self.description, device, maml_episode=False), batch_size=1, shuffle=False)
            elif key == 'train_warm_b':
                self.dataloaders['warmb_item2group'] = DataLoader(MovieLens1MmetaDataLoader(dataset_name, df, self.description, device, maml_episode=False), batch_size=1, shuffle=False)
            elif key == 'train_warm_c':
                self.dataloaders['warmc_item2group'] = DataLoader(MovieLens1MmetaDataLoader(dataset_name, df, self.description, device, maml_episode=False), batch_size=1, shuffle=False)
            elif key == 'test':
                self.dataloaders['test_item2group'] = DataLoader(MovieLens1MmetaDataLoader(dataset_name, df, self.description, device, maml_episode=False), batch_size=1, shuffle=False)



        self.keys = list(self.dataloaders.keys())
        self.item_features = ['item_id', 'year', 'title', 'genres']
                                

    def __getitem__(self, name):
        assert name in self.keys, '{} not in keys of datasets'.format(name)
        return self.dataloaders[name]

class Movielens1MgmebaseDataset(Dataset):
    """
    Load a base Movielens Dataset 
    """
    def __init__(self, dataset_name, df, description, device, maml_episode=False):
        super(Movielens1MgmebaseDataset, self).__init__()
        self.dataset_name = dataset_name
        self.df = df
        self.length = len(df)
        self.name2array = {name: torch.from_numpy(np.array(list(df[name])).reshape([self.length, -1])).to(device) \
                                        for name in df.columns}
        self.format(description)
        self.features = [name for name in df.columns if name != 'rating']
            
        self.label = 'rating'

        self.maml_episode = maml_episode
        if maml_episode:
            user_count = self.df.groupby(['user_id']).size().reset_index(name='user_count')
            self.final_index = user_count[(user_count.user_count > 13) & (user_count.user_count < 100)].user_id.unique()

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
        if self.maml_episode:
            user_id = self.final_index[index]
            index = self.df[self.df.user_id == user_id].index.to_list()
            random.shuffle(index)
        return {name: self.name2array[name][index] for name in self.features}, \
                self.name2array[self.label][index].squeeze()

    def __len__(self):
        if self.maml_episode:
            return len(self.final_index)
        return self.length

class Movielens1MgmebaseDataset_ngb(Dataset):
    """
    Load a base Movielens Dataset 
    """
    def __init__(self, dataset_name, df, description, device, maml_episode=False):
        super(Movielens1MgmebaseDataset_ngb, self).__init__()
        self.dataset_name = dataset_name
        self.df = df
        self.length = len(df)
        #print(self.length)
        #print(df['year'][1])
        self.name2array = {name: torch.from_numpy(np.array(list(df[name])).reshape([self.length, -1])).to(device) \
                                        for name in df.columns}
        #print(self.name2array['year'][0])
        self.format(description)
        #print(self.name2array['year'][1])
        self.features_ngb = [name for name in df.columns if name != 'rating']
        self.features = ['user_id', 'gender', 'age', 'occupation', 'item_id', 'year', 'title', 'genres']
            
        self.label = 'rating'

        self.maml_episode = maml_episode
        if maml_episode:
            user_count = self.df.groupby(['user_id']).size().reset_index(name='user_count')
            self.final_index = user_count[(user_count.user_count > 13) & (user_count.user_count < 100)].user_id.unique()

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
        if self.maml_episode:
            user_id = self.final_index[index]
            index = self.df[self.df.user_id == user_id].index.to_list()
            random.shuffle(index)
        return {name: self.name2array[name][index] for name in self.features}, \
                {name: self.name2array[name][index] for name in self.features_ngb}, self.name2array[self.label][index].squeeze()

    def __len__(self):
        if self.maml_episode:
            return len(self.final_index)
        return self.length


class MovieLens1MgmeColdStartDataLoader(object):
    """
    Load all splitted MovieLens 1M Dataset for cold start setting

    :param dataset_path: MovieLens dataset path
    """

    def __init__(self, dataset_name, dataset_path, device, bsz=32, shuffle=True, maml_episode=False):
        super(MovieLens1MgmeColdStartDataLoader, self).__init__()
        assert os.path.exists(dataset_path), '{} does not exist'.format(dataset_path)
        with open(dataset_path, 'rb+') as f:
            data = pickle.load(f)
        self.dataset_name = dataset_name
        self.dataloaders = {}
        self.description = data['description']
        for key, df in data.items():
            if key == 'description':
                continue
            if ('gme_a' == key) or ('gme_b' == key) or ('gme_train_warm_a' == key) or ('test_test' == key):
                self.dataloaders[key] = DataLoader(Movielens1MgmebaseDataset_ngb(dataset_name, df, self.description, device, maml_episode=False), batch_size=60, shuffle=False)
            elif ('train_warm_a' == key) or ('train_warm_b' == key) or ('train_warm_c' == key):
                self.dataloaders[key] = DataLoader(Movielens1MgmebaseDataset(dataset_name, df, self.description, device, maml_episode=False), batch_size=bsz, shuffle=True)
            else:
                self.dataloaders[key] = DataLoader(Movielens1MgmebaseDataset(dataset_name, df, self.description, device, maml_episode=False), batch_size=bsz, shuffle=True)
        self.keys = list(self.dataloaders.keys())
        self.item_features = ['year', 'title', 'genres']
                                

    def __getitem__(self, name):
        assert name in self.keys, '{} not in keys of datasets'.format(name)
        return self.dataloaders[name]