import json
import os

import torch
import numpy as np
import torch.nn.functional as F

from scipy.io import loadmat
from torch.utils.data import Dataset, BatchSampler
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import IWSLT2017
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from typing import List

translation_root = 'data/translation'
spacy_lang = {
    'de': 'de_core_news_sm',
    'en': 'en_core_web_sm',
    'it': 'it_core_news_sm',
}
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

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


def _iwslt2017trans_data_collate(batch):
    X = pad_sequence([b[0] for b in batch], padding_value=PAD_IDX)
    y = pad_sequence([b[1] for b in batch], padding_value=PAD_IDX)
    return X, y


# Part of this dataset script use information from: https://pytorch.org/tutorials/beginner/translation_transformer.html
class IWSLT2017TransDataset(Dataset):
    def __init__(self, src_lang='en', tgt_lang='de', dataset_type='train', train_subset=25000, subset=25000, max_len=50, min_freq=5, load_from_file=True):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_tokenizer = get_tokenizer('spacy', language=spacy_lang[self.src_lang])
        self.tgt_tokenizer = get_tokenizer('spacy', language=spacy_lang[self.tgt_lang])

        # build vocab
        language_index = {self.src_lang: 0, self.tgt_lang: 1}
        def yield_tokens(data_iter, tokenizer, lang_idx):
            for data_sample in data_iter:
                yield tokenizer(data_sample[lang_idx])
        vocab_path = os.path.join(translation_root, 'IWSLT2017/vocab')
        os.makedirs(vocab_path, exist_ok=True)

        print(f"building {src_lang} vocab...")
        src_vocab_fname = os.path.join(vocab_path, f'{src_lang}_src_{train_subset}.pt')
        if load_from_file and os.path.exists(src_vocab_fname):
            print(f"found previously built {src_lang} vocab, loading...")
            self.src_vocab = torch.load(src_vocab_fname)
        else:
            if train_subset is None:
                train_set = IWSLT2017(root=translation_root, split='train', language_pair=(self.src_lang, self.tgt_lang))
            else:
                print(f"taking a {train_subset} subset of train data to build vocab...")
                train_iter = IWSLT2017(root=translation_root, split='train', language_pair=(self.src_lang, self.tgt_lang))
                train_set = [next(train_iter) for _ in range(subset)]
            self.src_vocab = build_vocab_from_iterator(yield_tokens(train_set, self.src_tokenizer, language_index[self.src_lang]),
                                                       min_freq=min_freq,
                                                       specials=special_symbols,
                                                       special_first=True)
            torch.save(self.src_vocab, src_vocab_fname)
        print(f"there are {len(self.src_vocab)} tokens in {src_lang} vocab")

        print(f"building {tgt_lang} vocab...")
        tgt_vocab_fname = os.path.join(vocab_path, f'{tgt_lang}_tgt_{train_subset}.pt')
        if load_from_file and os.path.exists(tgt_vocab_fname):
            print(f"found previously built {tgt_lang} vocab, loading...")
            self.tgt_vocab = torch.load(tgt_vocab_fname)
        else:
            if train_subset is None:
                train_set = IWSLT2017(root=translation_root, split='train', language_pair=(self.src_lang, self.tgt_lang))
            else:
                print(f"taking a {train_subset} subset of train data to build vocab...")
                train_iter = IWSLT2017(root=translation_root, split='train', language_pair=(self.src_lang, self.tgt_lang))
                train_set = [next(train_iter) for _ in range(subset)]
            self.tgt_vocab = build_vocab_from_iterator(yield_tokens(train_set, self.tgt_tokenizer, language_index[self.tgt_lang]),
                                                       min_freq=min_freq,
                                                       specials=special_symbols,
                                                       special_first=True)
            torch.save(self.tgt_vocab, tgt_vocab_fname)
        print(f"there are {len(self.tgt_vocab)} tokens in {tgt_lang} vocab")

        self.src_vocab.set_default_index(UNK_IDX)
        self.tgt_vocab.set_default_index(UNK_IDX)

        # build dataset
        print(f"building translation {dataset_type} {src_lang}->{tgt_lang} dataset...")
        data_path = os.path.join(translation_root, 'IWSLT2017/translation')
        os.makedirs(data_path, exist_ok=True)
        preprocessed_fname = os.path.join(data_path, f'{dataset_type}_{src_lang}_to_{tgt_lang}_{train_subset}.json')
        if load_from_file and os.path.exists(preprocessed_fname):
            print(f"found preprocessed {dataset_type} {src_lang}->{tgt_lang} dataset, loading...")
            with open(preprocessed_fname, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = []
            data_iter = IWSLT2017(root=translation_root, split=dataset_type, language_pair=(self.src_lang, self.tgt_lang))
            for src_text, tgt_text in data_iter:
                src_ids = self.convert_src_text_to_ids(src_text)
                tgt_ids = self.convert_tgt_text_to_ids(tgt_text)
                self.data.append([src_ids, tgt_ids])
            with open(preprocessed_fname, 'w') as f:
                json.dump(self.data, f)

        # subset the dataset as needed
        if subset is not None:
            print(f"taking {subset} subset of full data...")
            self.data = self.data[:subset]

        # truncate to max length
        if max_len is not None:
            print(f"truncating all sequences to have max {max_len} tokens...")
            for idx in range(len(self.data)):
                # source
                if len(self.data[idx][0]) > max_len:
                    self.data[idx][0] = self.data[idx][0][:max_len - 1] + [EOS_IDX]
                elif len(self.data[idx][0]) < max_len:
                    # add padding for batches
                    padding = [PAD_IDX] * (max_len - len(self.data[idx][0]))
                    self.data[idx][0] = self.data[idx][0][:-1] + padding + [EOS_IDX]
                # target
                if len(self.data[idx][1]) > max_len:
                    self.data[idx][1] = self.data[idx][1][:max_len - 1] + [EOS_IDX]
                elif len(self.data[idx][1]) < max_len:
                    # add padding for batches
                    padding = [PAD_IDX] * (max_len - len(self.data[idx][1]))
                    self.data[idx][1] = self.data[idx][1][:-1] + padding + [EOS_IDX]

        self.max_src_len = max(map(lambda ex: len(ex[0]), self.data))
        self.max_tgt_len = max(map(lambda ex: len(ex[1]), self.data))

    def get_batch(self, batch_size, batch_num):
        # calculate start and end idxs of current batch
        start_idx = batch_num * batch_size
        end_idx = start_idx + batch_size
        if end_idx >= len(self.data):
            end_idx = len(self.data)

        # get src and tgt batches
        src_batch = []
        tgt_batch = []
        for idx in range(start_idx, end_idx):
            src_batch.append(self.data[idx][0])
            tgt_batch.append(self.data[idx][1])
        
        return torch.transpose(torch.tensor(src_batch, dtype=torch.int64),0,1), \
            torch.transpose(torch.tensor(tgt_batch, dtype=torch.int64),0,1)

    def _convert_text_to_ids(self, text, tokenizer, vocab_indexer):
        text = text.rstrip('\n')
        tokens = tokenizer(text)
        token_ids = vocab_indexer(tokens)
        full_ids = [BOS_IDX] + token_ids + [EOS_IDX]
        return full_ids

    def convert_src_text_to_ids(self, text):
        return self._convert_text_to_ids(text, self.src_tokenizer, self.src_vocab)

    def convert_tgt_text_to_ids(self, text):
        return self._convert_text_to_ids(text, self.tgt_tokenizer, self.tgt_vocab)

    def _convert_ids_to_tokens(self, ids: List[int], vocab_indexer):
        if ids[0] == BOS_IDX:
            ids = ids[1:]
        if ids[-1] == EOS_IDX:
            ids = ids[:-1]
        tokens = vocab_indexer.lookup_tokens(ids)
        return tokens

    def convert_src_ids_to_tokens(self, ids: List[int]):
        return self._convert_ids_to_tokens(ids, self.src_vocab)

    def convert_src_ids_to_text(self, ids: List[int]):
        return " ".join(self.convert_src_ids_to_tokens(ids))

    def convert_tgt_ids_to_tokens(self, ids: List[int]):
        return self._convert_ids_to_tokens(ids, self.tgt_vocab)

    def convert_tgt_ids_to_text(self, ids: List[int]):
        return " ".join(self.convert_tgt_ids_to_tokens(ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src = torch.tensor(src, dtype=torch.int64)
        tgt = torch.tensor(tgt, dtype=torch.int64)
        return src, tgt


def _compute_nll(logits, target):
    # computes the negative log-likelihood of a tensor given a target
    log_p = F.logsigmoid(logits) * target
    nll = torch.neg(torch.sum(log_p))
    return nll


def _compute_bleu(model, data, device, subset=None):
    bleu_scores = []
    if subset is None or subset > len(data):
        subset = len(data)
    print(f"calculating bleu for {subset} examples...")
    for idx in tqdm(range(subset)):
        src, tgt = data[idx]
        src = src.unsqueeze(1).to(device)
        pred, score = model.infer(src)
        pred = pred.squeeze(1).detach().cpu()
        ref = [tgt.tolist()]
        candidate = pred.tolist()
        score = sentence_bleu(ref, candidate)
        bleu_scores.append(score)
    return sum(bleu_scores) / len(bleu_scores)


def save(save_dir, epoch, model, optim, train_losses, valid_losses, bleu_score=None):
    # states_dict = {
    #     "epoch": epoch,
    #     "train_loss": losses[0],
    #     "valid_loss": losses[1],
    #     "train_metric": metrics[0],
    #     "valid_metric": metrics[1],
    # }
    states_dict = {
        "epoch": epoch,
        "train_loss": train_losses,
        "valid_loss": valid_losses
    }
    if bleu_score is not None:
        states_dict['bleu_score'] = bleu_score
    with open(os.path.join(save_dir, 'states.json'), 'w') as f:
        json.dump(states_dict, f)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
    torch.save(optim.state_dict(), os.path.join(save_dir, 'optim.pt'))


def load(load_dir, model, optim, device):
    with open(os.path.join(load_dir, 'states.json'), 'r') as f:
        states_dict = json.load(f)
    if device.type == 'cpu':
        model.load_state_dict(torch.load(os.path.join(load_dir, 'model.pt'), map_location=('cpu')))
    else:
        model.load_state_dict(torch.load(os.path.join(load_dir, 'model.pt')))
    optim.load_state_dict(torch.load(os.path.join(load_dir, 'optim.pt')))
    epoch = states_dict['epoch'] + 1
    losses = [states_dict['train_loss'], states_dict['valid_loss']]
    # metrics = [states_dict['train_metric'], states_dict['valid_metric']]
    return epoch, model, optim, losses#, metrics