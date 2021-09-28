# -*- coding: utf-8 -*-
"""

Automatically generated by Colaboratory.

This script uses code from the below sources:
https://pytorch.org/tutorials/beginner/translation_transformer.html
https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
"""

# !python -m spacy download en_core_web_sm
# !python -m spacy download de_core_news_sm

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import IWSLT2017, Multi30k
from typing import Iterable, List
import random
import matplotlib.pyplot as plt
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import numpy as np
# import time
from torch.nn.utils.rnn import pad_sequence
from timeit import default_timer as timer
import pickle as pkl

import ipdb

SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'de'

new_IWSLT2017 = []
new_IWSLT2017_valid = []
i = 0
for data in IWSLT2017(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE)):
    # new_IWSLT2017.append(data)
  if i < 25000:
    new_IWSLT2017.append(data)
  else:
    break
  i = i + 1

i=0
for data in IWSLT2017(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE)):
    # new_IWSLT2017.append(data)
  if i < 5000:
    new_IWSLT2017_valid.append(data)
  else:
    break
  i = i + 1

# for data in Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE)):
#     # new_IWSLT2017.append(data)
#   new_IWSLT2017.append(data)


# i=0
# for data in Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE)):
#     # new_IWSLT2017.append(data)
#   new_IWSLT2017_valid.append(data)

# Place-holders
token_transform = {}
vocab_transform = {}

# ipdb.set_trace()


# Create source and target language tokenizer. Make sure to install the dependencies.
# pip install -U spacy
# python -m spacy download en_core_web_sm
# python -m spacy download de_core_news_sm
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')


# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

# ipdb.set_trace()

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    train_iter = new_IWSLT2017
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=5,
                                                    specials=special_symbols,
                                                    special_first=True)

# ipdb.set_trace()

# Set UNK_IDX as the default index. This index is returned when the token is not found.
# If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)

# ipdb.set_trace()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# torch.cuda.empty_cache()
# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell


# Seq2Seq Network
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs


torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

BATCH_SIZE = 64


INPUT_DIM = SRC_VOCAB_SIZE #len(train_data.src_vocab)
OUTPUT_DIM = TGT_VOCAB_SIZE #len(train_data.tgt_vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

transformer = Seq2Seq(enc, dec, device).to(device)


for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# optimizer = torch.optim.Adam(transformer.parameters())#(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

# ipdb.set_trace()

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
# ipdb.set_trace()

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor

# ipdb.set_trace()

# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))
    # ipdb.set_trace()

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

from torch.utils.data import DataLoader

def train_epoch(model, optimizer, data_iterator):
    # ipdb.set_trace()
    model.train()
    losses = 0
    train_iter = data_iterator #new_IWSLT2017
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    # ipdb.set_trace()

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        logits = model(src, tgt)

        optimizer.zero_grad()

        # ipdb.set_trace()
        tgt_out = tgt

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)

def valid_epoch(model, optimizer, data_iterator):
    # ipdb.set_trace()
    model.eval()
    losses = 0
    valid_iter = data_iterator #new_IWSLT2017
    valid_dataloader = DataLoader(valid_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    # ipdb.set_trace()

    for src, tgt in valid_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        logits = model(src, tgt, teacher_forcing_ratio=0)
        
        tgt_out = tgt

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        losses += loss.item()

    return losses / len(valid_dataloader)


NUM_EPOCHS = 20

train_losses = []
valid_losses = []
epochs_log = []
for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer, new_IWSLT2017)
    valid_loss = valid_epoch(transformer, optimizer, new_IWSLT2017_valid)
    end_time = timer()
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    print((f"Epoch: {epoch}, Validation loss: {valid_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    epochs_log.append(epoch)
print(SRC_VOCAB_SIZE)
print(TGT_VOCAB_SIZE)



out_strs=[]
def valid_translate(model: torch.nn.Module):
    model.eval()
    valid_iter = new_IWSLT2017_valid
    valid_dataloader = DataLoader(valid_iter, batch_size=1, collate_fn=collate_fn)
    # ipdb.set_trace()

    for src, tgt in valid_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        # logits = model(src, tgt)
        output = model.forward(src,tgt, teacher_forcing_ratio=0)
        topv,topi = output.data.topk(1) 
        tgt_tokens = topi
        # out_strs.append(" ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(np.squeeze(tgt_tokens.cpu().detach().numpy()).astype(int)))).replace("<bos>", "").replace("<eos>", ""))
        out_strs.append(" ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens((np.squeeze((tgt_tokens.cpu().detach().numpy()).astype(int))).tolist())).replace("<bos>", "").replace("<eos>", ""))
    return out_strs

out_sentences = valid_translate(model=transformer)

# ipdb.set_trace()

torch.save(transformer.state_dict(), 'seq2seq_epochs'+str(NUM_EPOCHS)+'_'+SRC_LANGUAGE+'_'+TGT_LANGUAGE+'.pt')
print(out_sentences)

data = {"train_losses": train_losses, "validation_losses":valid_losses, "epochs":epochs_log}
pkl.dump(data,open("seq2seq_epochs"+str(NUM_EPOCHS)+'_'+SRC_LANGUAGE+'_'+TGT_LANGUAGE+".p","wb"))

l1, = plt.plot(train_losses, color='skyblue', linewidth=2, label='training losses')
l2, = plt.plot(valid_losses, color='red', linewidth=2, label='validation losses')
plt.ylabel('cross entropy loss')
plt.xlabel('epochs')
plt.xticks(np.arange(0, NUM_EPOCHS, 5))
plt.legend(handles = [l1,l2])
plt.title('Train and valid losses');# (y offset is for plotting purposes)')
plt.savefig('seq2seq_losses_epochs'+str(NUM_EPOCHS)+'_'+SRC_LANGUAGE+'_'+TGT_LANGUAGE+'.png')
plt.close()


# path = "./drive/MyDrive/Colab Notebooks/transformer_en_de.ipynb"
# torch.save(transformer.state_dict(), path)

# a =[]
# b =[1,2,3]
# a = b
# print(a)