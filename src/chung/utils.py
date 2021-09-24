import os

import torch
import numpy as np

from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

# Source: https://github.com/locuslab/TCN/blob/master/TCN/poly_music/utils.py
def music_dataloader(dataset, prefix="."):
    if dataset == "JSB":
        print('loading JSB data...')
        data = loadmat(os.path.join(prefix, 'data/music/JSB_Chorales.mat'))
    elif dataset == "Muse":
        print('loading Muse data...')
        data = loadmat(os.path.join(prefix, 'data/music/MuseData.mat'))
    elif dataset == "Nott":
        print('loading Nott data...')
        data = loadmat(os.path.join(prefix, 'data/music/Nottingham.mat'))
    elif dataset == "Piano":
        print('loading Piano data...')
        data = loadmat(os.path.join(prefix, 'data/music/Piano_midi.mat'))
    else:
        raise ValueError(f"No data for '{dataset}' found. Possible datasets from: JSB, Muse, Nott, Piano.")

    X_train = data['traindata'][0]
    X_valid = data['validdata'][0]
    X_test = data['testdata'][0]

    for data in [X_train, X_valid, X_test]:
        for i in range(len(data)):
            data[i] = torch.Tensor(data[i].astype(np.float64))

    return X_train, X_valid, X_test

class MusicDataset(Dataset):
    def __init__(self, dataset_name, dataset_type='train'):
        data_index = {'train': 0, 'valid': 1, 'test': 2}
        idx = data_index[dataset_type]
        if dataset_name == 'all':
            self.data = np.array([])
            for dataset in ("JSB", "Muse", "Nott", "Piano"):
                self.data = np.append(self.data, music_dataloader(dataset)[idx])
        else:
            self.data = music_dataloader(dataset_name)[idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx][:-1]
        y = self.data[idx][1:]
        return X, y
