import json
import os

import torch
import numpy as np
import torch.nn.functional as F

from scipy.io import loadmat
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

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


def _music_data_collate(batch):
    X = pad_sequence([b[0] for b in batch])
    y = pad_sequence([b[1] for b in batch])
    return X, y


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


def _compute_nll(logits, target):
    # computes the negative log-likelihood of a tensor given a target
    log_p = F.logsigmoid(logits) * target
    nll = torch.neg(torch.sum(log_p))
    return nll


def save(save_dir, epoch, model, optim, losses, metrics):
    states_dict = {
        "epoch": epoch,
        "train_loss": losses[0],
        "valid_loss": losses[1],
        "train_metric": metrics[0],
        "valid_metric": metrics[1],
        }
    with open(os.path.join(save_dir, 'states.json'), 'w') as f:
        json.dump(states_dict, f)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
    torch.save(optim.state_dict(), os.path.join(save_dir, 'optim.pt'))


def load(load_dir, model, optim):
    with open(os.path.join(load_dir, 'states.json'), 'r') as f:
        states_dict = json.load(f)
    model.load_state_dict(torch.load(os.path.join(load_dir, 'model.pt')))
    optim.load_state_dict(torch.load(os.path.join(load_dir, 'optim.pt')))
    epoch = states_dict['epoch'] + 1
    losses = [states_dict['train_loss'], states_dict['valid_loss']]
    metrics = [states_dict['train_metric'], states_dict['valid_metric']]
    return epoch, model, optim, losses, metrics
