import io
import json
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

from models import MusicLM, S2STrans
from utils import (
    MusicDataset, IWSLT2017TransDataset,
    _music_data_collate, _iwslt2017trans_data_collate,
    _compute_nll, _compute_bleu,
    save, load,
    UNK_IDX, PAD_IDX, EOS_IDX
)

SEED = 42


def initialize(args):
    # seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("------------ Configs ------------")

    if args.task == 'music_lm':
        train_data = MusicDataset(args.data, dataset_type='train')
        valid_data = MusicDataset(args.data, dataset_type='valid')
        model = MusicLM(rnn_type=args.rnn, num_layers=args.num_layers).to(device)
        print("model summary: ")
        summary(model, torch.zeros((200, args.bs, 88)).to(device))  # Assuming 200 sequence length on average, 88 features
        print("rnn: ", args.rnn)
        print("num layers:", args.num_layers)
        print("data: ", args.data)
    elif args.task == 'translate':
        data_args = args.data.split("_")
        if data_args[0] == "iwslt":
            train_data = IWSLT2017TransDataset(src_lang=data_args[1],
                    tgt_lang=data_args[2], train_subset=args.train_subset,
                    subset=args.train_subset, max_len=args.max_len, dataset_type='train',
                    min_freq=args.min_freq,
                    load_from_file=not args.rebuild_data)
            valid_data = IWSLT2017TransDataset(src_lang=data_args[1],
                    tgt_lang=data_args[2], train_subset=args.train_subset,
                    subset=args.valid_subset, max_len=args.max_len, dataset_type='valid',
                    min_freq=args.min_freq,
                    load_from_file=not args.rebuild_data)
            model = S2STrans(src_vocab_size=len(train_data.src_vocab),
                             tgt_vocab_size=len(train_data.tgt_vocab),
                             embedding_dim=args.embed_dim,
                             rnn_type=args.rnn,
                             num_layers=args.num_layers).to(device)
            print("model summary: ")
            summary(model, torch.zeros((args.bs, args.max_len), dtype=torch.int64).to(device),
                    torch.zeros((args.bs, args.max_len), dtype=torch.int64).to(device))
            print("rnn: ", args.rnn)
            print("num layers:", args.num_layers)
            print("embedding dim: ", args.embed_dim)
            print("data: ", args.data)
            print("max seq tokens: ", args.max_len)
            print("train subset: ", args.train_subset)
            print("valid subset: ", args.valid_subset)
        else:
            raise NotImplementedError("Datasets other than IWSLT2017 for translation are not implemented.")

    print("optimizer: ", args.optim)
    if args.optim == 'rmsprop':
        optim = RMSprop(model.parameters(), lr=args.lr, momentum=args.mtm, weight_decay=args.wd)
        print("learning rate: ", args.lr)
        print("momentum: ", args.mtm)
        print("weight decay: ", args.wd)
    elif args.optim == 'adamw':
        optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        print("learning rate: ", args.lr)
        print("weight decay: ", args.wd)
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
    elif args.loss == 'ce':
        weight = torch.ones(len(train_data.tgt_vocab))
        weight[UNK_IDX] = 0.1  # reduce <unk> outputs
        weight[EOS_IDX] = 0.1  # reduce early stopping
        weight = weight.to(device)
        criterion = nn.CrossEntropyLoss(weight=weight, reduction='mean', ignore_index=PAD_IDX)

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


def tb_add_trans_ex(logger, data, model, x, y, device, epoch, save_dir=None):
    model.eval()
    x_b = x.unsqueeze(1).to(device)
    y_hat, y_hat_score = model.infer(x_b)
    y_hat = y_hat.squeeze(1).detach().cpu()
    y_json = {'ids': y.tolist(), 'src_text': data.convert_src_ids_to_text(x.tolist()), 'tgt_text': data.convert_tgt_ids_to_text(y.tolist())}
    y_hat_json = {'ids': y_hat.tolist(), 'src_text': data.convert_src_ids_to_text(x.tolist()), 'tgt_text': data.convert_tgt_ids_to_text(y_hat.tolist()), 'nll': y_hat_score.item()}
    logger.add_text('src_text', y_json['src_text'], epoch)
    logger.add_text('pred_text', y_hat_json['tgt_text'], epoch)
    logger.add_text('gold_text', y_json['tgt_text'], epoch)
    logger.add_scalar('pred_text_nll', y_hat_score, epoch)
    if save_dir:
        with open(os.path.join(save_dir, f'{epoch}.json'), 'w') as f:
            json.dump(y_hat_json, f)
        with open(os.path.join(save_dir, 'gold.json'), 'w') as f:
            json.dump(y_json, f)


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
                #torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip, error_if_nonfinite=True)
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


def train_trans(args):
    (train, valid, optim, model, criterion, device,
        epochs, losses, metrics, step, log_dir, save_dir) = initialize(args)
    train_logger, valid_logger = None, None
    train_logger = tb.SummaryWriter(os.path.join(log_dir, 'train'), flush_secs=1)
    valid_logger = tb.SummaryWriter(os.path.join(log_dir, 'valid'), flush_secs=1)

    train_loader = DataLoader(train, batch_size=args.bs, shuffle=True, num_workers=4, collate_fn=_iwslt2017trans_data_collate)
    valid_loader = DataLoader(valid, batch_size=args.bs, shuffle=False, num_workers=4, collate_fn=_iwslt2017trans_data_collate)

    train_ex_idx = 42  # random.randint(0, len(train) - 1)
    val_ex_idx = 42  # random.randint(0, len(valid) - 1)

    for epoch in tqdm(epochs):
        # train
        train_loss, train_nll, train_nll_count = [], [], []
        for inp, target in train_loader:
            model.train()
            inp, target = inp.to(device), target.to(device)

            logits = model(inp, target[:-1])
            if args.loss != 'ce':
                raise NotImplementedError("For translation, only Cross Entropy ('ce') loss is used.")
            logits_l = logits.permute(1, 2, 0)
            target_l = target[1:].permute(1, 0)
            loss_val = criterion(logits_l, target_l)

            optim.zero_grad()
            loss_val.backward()
            if args.clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optim.step()

            model.eval()
            logits = logits.detach().cpu()
            target_one = F.one_hot(target[1:].detach().cpu(), num_classes=len(train.tgt_vocab))
            target_mask = (target[1:].detach().cpu() != PAD_IDX).unsqueeze(-1)
            target_mask = target_mask.repeat(1, 1, len(train.tgt_vocab))
            target_one = target_mask * target_one
            nll = _compute_nll(logits, target_one)
            count = logits.shape[0] * logits.shape[1]
            train_loss.append(loss_val.detach().cpu().numpy())
            train_nll.append(nll.numpy())
            train_nll_count.append(count)
            if train_logger:
                train_logger.add_scalar('loss', loss_val, step)
                train_logger.add_scalar('nll', nll / count, step)
            step += 1

        if train_logger:
            ex_x, ex_y = train[train_ex_idx]
            text_dir = os.path.join(save_dir, 'text_train')
            os.makedirs(text_dir, exist_ok=True)
            tb_add_trans_ex(train_logger, train, model, ex_x, ex_y, device, epoch, text_dir)

        # valid
        model.eval()
        val_loss, val_nll, val_nll_count = [], [], []
        for inp, target in valid_loader:
            inp, target = inp.to(device), target.to(device)

            logits = model(inp, target[:-1])
            if args.loss != 'ce':
                raise NotImplementedError("For translation, only Cross Entropy ('ce') loss is used.")
            logits_l = logits.permute(1, 2, 0)
            target_l = target[1:].permute(1, 0)
            loss_val = criterion(logits_l, target_l)

            logits = logits.detach().cpu()
            target_one = F.one_hot(target[1:].detach().cpu(), num_classes=len(train.tgt_vocab))
            target_mask = (target[1:].detach().cpu() != PAD_IDX).unsqueeze(-1)
            target_mask = target_mask.repeat(1, 1, len(train.tgt_vocab))
            target_one = target_mask * target_one
            nll = _compute_nll(logits, target_one)
            count = logits.shape[0] * logits.shape[1]
            val_loss.append(loss_val.detach().cpu().numpy())
            val_nll.append(nll.numpy())
            val_nll_count.append(count)

        if valid_logger:
            ex_x, ex_y = valid[val_ex_idx]
            text_dir = os.path.join(save_dir, 'text_valid')
            os.makedirs(text_dir, exist_ok=True)
            tb_add_trans_ex(valid_logger, valid, model, ex_x, ex_y, device, epoch, text_dir)

        # compute bleu score of a subset of the validation set
        valid_bleu = _compute_bleu(model, valid, device, subset=20)

        # log and save
        mean_train_loss = sum(train_loss) / len(train_loss)
        mean_train_nll = sum(train_nll) / sum(train_nll_count)
        mean_val_loss = sum(val_loss) / len(val_loss)
        mean_val_nll = sum(val_nll) / sum(val_nll_count)
        if valid_logger:
            valid_logger.add_scalar('loss', mean_val_loss, step)
            valid_logger.add_scalar('nll', mean_val_nll, step)
            valid_logger.add_scalar('bleu', valid_bleu, step)
        losses[0].append(mean_train_loss)
        losses[1].append(mean_val_loss)
        metrics[0].append(mean_train_nll)
        metrics[1].append(mean_val_nll)
        if args.verbose:
            print("[{}] epoch {}: train_loss: {}, valid_loss: {}, train_nll: {}, valid_nll: {}, valid_bleu: {}".format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                epoch,
                mean_train_loss,
                mean_val_loss,
                mean_train_nll,
                mean_val_nll,
                valid_bleu))

        # save latest
        latest_dir = os.path.join(save_dir, "latest")
        os.makedirs(latest_dir, exist_ok=True)
        save(latest_dir, epoch, model, optim, losses, metrics, bleu_score=valid_bleu)

        # save per 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"saving at epoch {epoch}...")
            # compute full valid bleu per 50 epochs
            if (epoch + 1) % 50 == 0:
                valid_bleu = _compute_bleu(model, valid, device)
            epoch_dir = os.path.join(save_dir, f"{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            save(epoch_dir, epoch, model, optim, losses, metrics, bleu_score=valid_bleu)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # task type
    parser.add_argument('--task', type=str, default='music_lm', choices=['translate', 'music_lm'])

    # dataset
    parser.add_argument('--data', type=str, default='all', choices=['all', 'JSB', 'Muse', 'Piano', 'Nott', 'iwslt_en_de', 'iwslt_de_it', 'iwslt_it_en'])

    # dataset subset
    parser.add_argument('--train_subset', type=int, default=25000)

    # dataset subset
    parser.add_argument('--valid_subset', type=int, default=5000)

    # dataset rebuild
    parser.add_argument('--rebuild_data', action='store_true')

    # max sequence length for translation
    parser.add_argument('--max_len', type=int, default=50)

    # min witness of token to be added to vocab
    parser_add_argument('--min_freq', type=int, default=5)

    # model type
    parser.add_argument('--rnn', type=str, default='gru', choices=['lstm', 'gru', 'tanh'])

    # number of rnn layers
    parser.add_argument('--num_layers', type=int, default=1)

    # embedding dim
    parser.add_argument('--embed_dim', type=int, default=512)

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
    parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'kl', 'ce'])

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
        train_trans(args)
