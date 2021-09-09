import numpy as np
import torch
import random
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
import copy

def prepare_train_val_data(data, PS, batch_size, val_idx, k):
    data.df = data.df.sample(frac=1)
    #data.df.reset_index(drop=True, inplace=True)
    class_count = np.array([len(np.where(data.df['relations_id']==t)[0]) for t in np.unique(data.df['relations_id'])])
    weight = 1. / class_count

    val_data = copy.deepcopy(data)
    val_size = len(data) // k
    val_data.df = val_data.df[val_idx*val_size: (val_idx+1)*val_size]
    #val_data.df.reset_index(drop=True, inplace=True)

    data.df = data.df.drop(range(val_idx*val_size, (val_idx+1)*val_size))
    #data.df.reset_index(drop=True, inplace=True)

    samples_weight_val = np.array([weight[t] for t in val_data.df['relations_id']])
    samples_weight_val = torch.from_numpy(samples_weight_val)
    sampler_val = WeightedRandomSampler(samples_weight_val.type('torch.DoubleTensor'), len(samples_weight_val))
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, \
                        num_workers=0, collate_fn=PS, pin_memory=False, sampler = sampler_val)

    samples_weight_train = np.array([weight[t] for t in data.df['relations_id']])
    samples_weight_train = torch.from_numpy(samples_weight_train)
    sampler_train = WeightedRandomSampler(samples_weight_train.type('torch.DoubleTensor'), len(samples_weight_train))
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=False, \
                        num_workers=0, collate_fn=PS, pin_memory=False, sampler = sampler_train)

    return train_loader, val_loader