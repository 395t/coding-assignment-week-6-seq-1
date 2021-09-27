import json
import os
import random

import torch
import numpy as np

from tqdm import tqdm
from torchsummaryX import summary

from models import MusicLM, S2STrans
from utils import (
    MusicDataset, IWSLT2017TransDataset,
    _compute_nll, _compute_bleu,
    load,
)

SEED = 42


def initialize(args):
    # seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("------------ Configs ------------")

    # load directory
    save_dir = os.path.join("./exp", args.save_dir)
    log_dir = os.path.join("./log", args.save_dir)
    load_dir = os.path.join(save_dir, args.model_name)

    if args.task == 'music_lm':
        data = MusicDataset(args.data, dataset_type=args.data_type)
        model = MusicLM(rnn_type=args.rnn, num_layers=args.num_layers).to(device)
        states_dict, epoch, model, _, _, _ = load(load_dir, model)
        print("model summary: ")
        summary(model, torch.zeros((200, args.bs, 88)).to(device))  # Assuming 200 sequence length on average, 88 features
        print("rnn: ", args.rnn)
        print("num layers:", args.num_layers)
        print("data: ", args.data)
    elif args.task == 'translate':
        data_args = args.data.split("_")
        if data_args[0] == "iwslt":
            data = IWSLT2017TransDataset(src_lang=data_args[1],
                    tgt_lang=data_args[2], train_subset=args.train_subset,
                    subset=args.subset, max_len=args.max_len, dataset_type=args.data_type,
                    min_freq=args.min_freq,
                    load_from_file=not args.rebuild_data)
            model = S2STrans(src_vocab_size=len(data.src_vocab),
                             tgt_vocab_size=len(data.tgt_vocab),
                             embedding_dim=args.embed_dim,
                             rnn_type=args.rnn,
                             num_layers=args.num_layers).to(device)
            states_dict, epoch, model, _, _, _ = load(load_dir, model)
            print("model summary: ")
            summary(model, torch.zeros((args.bs, args.max_len), dtype=torch.int64).to(device),
                    torch.zeros((args.bs, args.max_len), dtype=torch.int64).to(device))
            print("rnn: ", args.rnn)
            print("num layers:", args.num_layers)
            print("embedding dim: ", args.embed_dim)
            print("data: ", args.data)
            print("max seq tokens: ", args.max_len)
            print("train subset: ", args.train_subset)
            print("data subset: ", args.subset)
        else:
            raise NotImplementedError("Datasets other than IWSLT2017 for translation are not implemented.")

    print("cuda: ", next(model.parameters()).is_cuda)
    print("---------------------------------")

    return model, data, device, states_dict, load_dir


def infer_dataset(args):
    model, data, device, states_dict, load_dir = initialize(args)
    model.eval()
    if args.task == "music_lm":
        nlls, nll_lens = [], []
        for idx in tqdm(range(len(data))):
            inp, tgt = data[idx]
            inp, tgt = inp.unsqueeze(1).to(device), tgt.unsqueeze(1).to(device)
            logits = model(inp)
            nll = _compute_nll(logits, tgt)
            nll = nll.detach().cpu().item()
            nlls.append(nll)
            nll_lens.append(inp.shape[0])
        score = sum(nlls) / sum(nll_lens)
        scores = [nll / nll_len for nll, nll_len in zip(nlls, nll_lens)]
        print(f"{args.data} {args.data_type} NLL score: ", score)
        states_dict[f"{args.data_type}_nll"] = score
        states_dict[f"{args.data_type}_nll_scores"] = scores
        print(f"saving state to {load_dir}")
        with open(os.path.join(load_dir, "states.json"), 'w') as f:
            json.dump(states_dict, f)
    elif args.task == "translate":
        score, scores = _compute_bleu(model, data, device)
        print(f"{args.data} {args.data_type} BLEU score: ", score)
        states_dict[f"{args.data_type}_bleu"] = score
        states_dict[f"{args.data_type}_bleu_scores"] = scores
        print(f"saving state to {load_dir}")
        with open(os.path.join(load_dir, "states.json"), 'w') as f:
            json.dump(states_dict, f)
    else:
        raise NotImplementedError(f"{args.task} is not implemented. Possible tasks are: translate, music_lm")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # task type
    parser.add_argument('--task', type=str, default='music_lm', choices=['translate', 'music_lm'])

    # dataset
    parser.add_argument('--data', type=str, default='all', choices=['all', 'JSB', 'Muse', 'Piano', 'Nott', 'iwslt_en_de', 'iwslt_de_it', 'iwslt_it_en'])

    # dataset type
    parser.add_argument('--data_type', type=str, default='valid', choices=['train', 'valid', 'test'])

    # dataset training subset
    parser.add_argument('--train_subset', type=int, default=25000)

    # dataset subset
    parser.add_argument('--subset', type=int, default=None)

    # dataset rebuild
    parser.add_argument('--rebuild_data', action='store_true')

    # max sequence length for translation
    parser.add_argument('--max_len', type=int, default=50)

    # min witness of token to be added to vocab
    parser.add_argument('--min_freq', type=int, default=5)

    # model type
    parser.add_argument('--rnn', type=str, default='gru', choices=['lstm', 'gru', 'tanh'])

    # number of rnn layers
    parser.add_argument('--num_layers', type=int, default=1)

    # embedding dim
    parser.add_argument('--embed_dim', type=int, default=512)

    # batch size
    parser.add_argument('--bs', type=int, default=1)

    # save path
    parser.add_argument('--save_dir', type=str, default="music/all/gru")

    # model name for inference
    parser.add_argument('--model_name', type=str, default="latest")

    args = parser.parse_args()
    infer_dataset(args)
