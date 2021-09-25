import io
import os
import random

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torch.utils.tensorboard as tb
import matplotlib.pyplot as plt

from datetime import datetime

from torch.optim import SGD, AdamW, RMSprop
from torch.utils.data import DataLoader, Subset
from torchsummaryX import summary
from tqdm import tqdm

from models import MusicLM
from utils import (
    MusicDataset, _music_data_collate, _compute_nll,
    save, load
)


def initialize(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("------------ Configs ------------")

    if args.task == 'music_lm':
        train_data = MusicDataset(args.data, dataset_type='train')
        valid_data = MusicDataset(args.data, dataset_type='valid')
        model = MusicLM(rnn_type=args.rnn).to(device)
        print("model summary: ")
        summary(model, torch.zeros((200, args.bs, 88)).to(device))  # Assuming 200 sequence length on average, 88 features
        print("rnn: ", args.rnn)
        print("data: ", args.data)
    elif args.task == 'translate':
        raise NotImplementedError

    print("optimizer: ", args.optim)
    if args.optim == 'rmsprop':
        optim = RMSprop(model.parameters(), lr=args.lr, momentum=args.mtm, weight_decay=args.wd)
        print("learning rate: ", args.lr)
        print("momentum: ", args.mtm)
        print("weight decay: ", args.wd)
    elif args.optim == 'adamw':
        optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        print("learning rate: ", args.lr)
        print("learning rate: ", args.lr)
    elif args.optim == 'sgd':
        optim = SGD(model.parameters(), lr=args.lr, momentum=args.mtm, weight_decay=args.wd)
        print("learning rate: ", args.lr)
        print("momentum: ", args.mtm)
        print("weight decay: ", args.wd)
    print("gradient clipping: ", args.clip)

    print("loss criterion: ", args.loss)
    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
    elif args.loss == 'kl':
        criterion = nn.KLDivLoss(reduction='batchmean')

    print("continue training: ", args.continue_training)
    save_dir = os.path.join("./exp", args.save_dir)
    log_dir = os.path.join("./log", args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print("save directory: ", save_dir)
    print("log directory: ", log_dir)

    start_ep = 0
    losses = [[], []]
    metrics = [[], []]
    if args.continue_training:
        load_dir = os.path.join(save_dir, args.model_name)
        print("loading from: ", load_dir)
        start_ep, model, optim, losses, metrics = load(load_dir, model, optim)
    print("starting epoch: ", start_ep)
    print("ending epoch: ", args.epochs)
    epochs = range(start_ep, args.epochs, 1)
    steps = len(train_data) // args.bs
    if len(train_data) % args.bs != 0:
        steps += 1  # Assuming drop_last=False
    print("steps per epoch: ", steps)
    global_step = start_ep * steps

    print("cuda: ", next(model.parameters()).is_cuda)
    print("---------------------------------")

    return (train_data, valid_data, optim, model, criterion, device, epochs,
            losses, metrics, global_step, log_dir, save_dir)


def gen_lm_fig(seq, epoch):
    ratio = seq.T.shape[0] // 8
    cols = seq.T.shape[0] // ratio
    rows = seq.T.shape[1] // ratio
    fig, ax = plt.subplots(figsize=(rows, cols))
    ax.imshow(seq.T, cmap='Greys', origin='lower')
    y_label_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    ax.set_yticks([3, 15, 27, 39, 51, 63, 75, 87])
    ax.set_yticklabels(y_label_list)
    ax.set_ylabel('Notes')
    ax.set_xlabel('Timestep')
    ax.set_title(f'Epoch {epoch}')
    return fig


def tb_add_lm_ex(logger, model, x, y, device, epoch, save_dir=None):
    model.eval()
    x = x.unsqueeze(1).to(device)
    y_hat = model.lm(x).squeeze(1).detach().cpu().numpy()
    fig = gen_lm_fig(y_hat, epoch)
    if save_dir:
        fig.savefig(os.path.join(save_dir, f'{epoch}.png'), format='png')
    logger.add_figure('pred_seq', fig, epoch)
    fig = gen_lm_fig(y, epoch)
    if save_dir:
        fig.savefig(os.path.join(save_dir, f'gold.png'), format='png')
    logger.add_figure('target_seq', fig, epoch)


def train_lm(args):
    (train, valid, optim, model, criterion, device,
        epochs, losses, metrics, step, log_dir, save_dir) = initialize(args)
    train_logger, valid_logger = None, None
    train_logger = tb.SummaryWriter(os.path.join(log_dir, 'train'), flush_secs=1)
    valid_logger = tb.SummaryWriter(os.path.join(log_dir, 'valid'), flush_secs=1)

    # train = Subset(train, range(100))

    train_loader = DataLoader(train, batch_size=args.bs, shuffle=True, num_workers=4, collate_fn=_music_data_collate)
    valid_loader = DataLoader(valid, batch_size=args.bs, shuffle=False, num_workers=4, collate_fn=_music_data_collate)

    train_ex_idx = 42  # random.randint(0, len(train) - 1)
    val_ex_idx = 42  # random.randint(0, len(valid) - 1)
    if args.data == 'Piano':
        val_ex_idx = 6

    for epoch in tqdm(epochs):
        # train
        train_loss, train_nll, train_nll_count = [], [], []
        for inp, target in train_loader:
            model.train()
            inp, target = inp.to(device), target.to(device)

            logits = model(inp)
            if args.loss == 'kl':
                logits = torch.transpose(logits, 0, 1)
                target = torch.transpose(target, 0, 1)
                logits_mod = F.logsigmoid(logits)
            else:
                logits_mod = logits
            loss_val = criterion(logits_mod, target)

            optim.zero_grad()
            loss_val.backward()
            if args.clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optim.step()

            model.eval()
            logits = logits.detach().cpu()
            target = target.detach().cpu()
            nll = _compute_nll(logits, target)
            count = inp.shape[0] * inp.shape[1]
            train_loss.append(loss_val.detach().cpu().numpy())
            train_nll.append(nll.numpy())
            train_nll_count.append(count)
            if train_logger:
                train_logger.add_scalar('loss', loss_val, step)
                train_logger.add_scalar('nll', nll / count, step)
            step += 1

        if train_logger:
            ex_x, ex_y = train[train_ex_idx]
            img_dir = os.path.join(save_dir, 'img_train')
            os.makedirs(img_dir, exist_ok=True)
            tb_add_lm_ex(train_logger, model, ex_x, ex_y, device, epoch, img_dir)

        # valid
        model.eval()
        val_loss, val_nll, val_nll_count = [], [], []
        for inp, target in valid_loader:
            inp, target = inp.to(device), target.to(device)

            logits = model(inp)
            if args.loss == 'kl':
                logits = torch.transpose(logits, 0, 1)
                target = torch.transpose(target, 0, 1)
                logits_mod = F.logsigmoid(logits)
            else:
                logits_mod = logits
            loss_val = criterion(logits_mod, target)

            logits = logits.detach().cpu()
            target = target.detach().cpu()
            nll = _compute_nll(logits, target)
            count = inp.shape[0] * inp.shape[1]
            val_loss.append(loss_val.detach().cpu().numpy())
            val_nll.append(nll.numpy())
            val_nll_count.append(count)

        if valid_logger:
            ex_x, ex_y = valid[val_ex_idx]
            img_dir = os.path.join(save_dir, 'img_valid')
            os.makedirs(img_dir, exist_ok=True)
            tb_add_lm_ex(valid_logger, model, ex_x, ex_y, device, epoch, img_dir)

        # log and save
        mean_train_loss = sum(train_loss) / len(train_loss)
        mean_train_nll = sum(train_nll) / sum(train_nll_count)
        mean_val_loss = sum(val_loss) / len(val_loss)
        mean_val_nll = sum(val_nll) / sum(val_nll_count)
        if valid_logger:
            valid_logger.add_scalar('loss', mean_val_loss, step)
            valid_logger.add_scalar('nll', mean_val_nll, step)
        losses[0].append(mean_train_loss)
        losses[1].append(mean_val_loss)
        metrics[0].append(mean_train_nll)
        metrics[1].append(mean_val_nll)
        if args.verbose:
            print("[{}] epoch {}: train_loss: {}, valid_loss: {}, train_nll: {}, valid_nll: {}".format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                epoch,
                mean_train_loss,
                mean_val_loss,
                mean_train_nll,
                mean_val_nll))

        # save latest
        latest_dir = os.path.join(save_dir, "latest")
        os.makedirs(latest_dir, exist_ok=True)
        save(latest_dir, epoch, model, optim, losses, metrics)

        # save per 100 epochs
        if (epoch + 1) % 100 == 0:
            epoch_dir = os.path.join(save_dir, f"{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            save(epoch_dir, epoch, model, optim, losses, metrics)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # task type
    parser.add_argument('--task', type=str, default='music_lm', choices=['translate', 'music_lm'])

    # dataset
    parser.add_argument('--data', type=str, default='all', choices=['all', 'JSB', 'Muse', 'Piano', 'Nott'])

    # model type
    parser.add_argument('--rnn', type=str, default='gru', choices=['lstm', 'gru', 'tanh'])

    # optimizer type
    parser.add_argument('--optim', type=str, default='rmsprop', choices=['sgd', 'adamw', 'rmsprop'])

    # learning rate
    parser.add_argument('--lr', type=float, default=1e-3)

    # momentum
    parser.add_argument('--mtm', type=float, default=0.9)

    # weight decay
    parser.add_argument('--wd', type=float, default=0.0)

    # gradient clipping
    parser.add_argument('--clip', type=float, default=1.0)

    # loss criterion
    parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'kl'])

    # missing arguments: weight noise, momentum clipping, LinearDecayOverEpoch

    # batch size
    parser.add_argument('--bs', type=int, default=1)

    # epochs
    parser.add_argument('--epochs', type=int, default=500)

    # save path
    parser.add_argument('--save_dir', type=str, default="music/all/gru")

    # model name for continue training
    parser.add_argument('--model_name', type=str, default="latest")

    # continue training
    parser.add_argument('-c', '--continue_training', action='store_true')

    # verbose
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    if args.task == 'music_lm':
        train_lm(args)
    elif args.task == 'translate':
        raise NotImplementedError
