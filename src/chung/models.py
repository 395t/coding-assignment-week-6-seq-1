import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import BOS_IDX, PAD_IDX, EOS_IDX


class MusicLM(nn.Module):
    def __init__(self, rnn_type="gru", input_size=88, hidden_size=None, num_layers=1,
            dropout=0, bidirectional=False):
        super().__init__()

        # Initialize RNN
        self.rnn = None
        if rnn_type == "gru":
            if hidden_size is None:
                hidden_size = 46
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                    num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        elif rnn_type == "lstm":
            if hidden_size is None:
                hidden_size = 38
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                    num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        elif rnn_type == "tanh":
            if hidden_size is None:
                hidden_size = 100
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, nonlinearity='tanh', dropout=dropout, bidirectional=bidirectional)
        else:
            raise ValueError("rnn_type must be one of the following: gru, lstm, tanh")

        # Initialize MLP
        self.mlp = nn.Linear((2 if bidirectional else 1) * hidden_size, input_size)

    def forward(self, X):
        output, _ = self.rnn(X)
        return self.mlp(output)

    def lm(self, X):
        return torch.sigmoid(self(X))


class S2STrans(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim=512, rnn_type="lstm", hidden_size=None, num_layers=1,
            dropout=0, bidirectional_enc=False, max_len=50):
        super().__init__()

        self.max_len = max_len
        self.bidirectional_enc = bidirectional_enc
        self.rnn_type = rnn_type

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, embedding_dim=embedding_dim, padding_idx=PAD_IDX)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embedding_dim=embedding_dim, padding_idx=PAD_IDX)

        # Initialize RNN
        self.enc, self.dec = None, None
        if rnn_type == "gru":
            if hidden_size is None:
                hidden_size = 587
            self.enc = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size,
                    num_layers=num_layers, dropout=dropout, bidirectional=bidirectional_enc)
            self.dec = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size,
                    num_layers=num_layers, dropout=dropout, bidirectional=False)
        elif rnn_type == "lstm":
            if hidden_size is None:
                hidden_size = 512
            self.enc = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                    num_layers=num_layers, dropout=dropout, bidirectional=bidirectional_enc)
            if self.bidirectional_enc:
                self.downsample_c = nn.Linear(hidden_size * 2, hidden_size)
            self.dec = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                    num_layers=num_layers, dropout=dropout, bidirectional=False)
        elif rnn_type == "tanh":
            if hidden_size is None:
                hidden_size = 958
            self.enc = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size,
                    num_layers=num_layers, nonlinearity='tanh', dropout=dropout, bidirectional=bidirectional_enc)
            self.dec = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size,
                    num_layers=num_layers, nonlinearity='tanh', dropout=dropout, bidirectional=False)
        else:
            raise ValueError("rnn_type must be one of the following: gru, lstm, tanh")
        if self.bidirectional_enc:
            self.downsample_h = nn.Linear(hidden_size * 2, hidden_size)

        # Initialize MLP
        self.mlp = nn.Linear(hidden_size, tgt_vocab_size)

    def forward(self, X, y_teacher):
        # reverse input
        X = torch.flip(X, [0])

        enc_output, enc_hid = self.enc(self.src_embedding(X))
        if self.bidirectional_enc:
            if self.rnn_type == 'lstm':
                enc_h = self.downsample_h(enc_hid[0])
                enc_c = self.downsample_c(enc_hid[1])
                enc_hid = (enc_h, enc_c)
            else:
                enc_hid = self.downsample_h(enc_hid)
        dec_output, _ = self.dec(self.tgt_embedding(y_teacher), enc_hid)
        return self.mlp(dec_output)

    def infer(self, X, beam_size=12):
        # reverse input
        X = torch.flip(X, [0])

        enc_output, enc_hid = self.enc(self.src_embedding(X))
        if self.bidirectional_enc:
            if self.rnn_type == 'lstm':
                enc_h = self.downsample_h(enc_hid[0])
                enc_c = self.downsample_c(enc_hid[1])
                enc_hid = (enc_h, enc_c)
            else:
                enc_hid = self.downsample_h(enc_hid)

        # beam search
        sequences = [(torch.LongTensor([[BOS_IDX]]).to(X), 0.0)]
        completed = []
        done = False
        for seq_idx in range(1, self.max_len):
            if sequences:
                new_sequences = []
                for seq, score in sequences:
                    _, dec_h = self.dec(self.tgt_embedding(seq), enc_hid)
                    if self.rnn_type == 'lstm':
                        dec_h = dec_h[0]
                    logits = self.mlp(dec_h[-1])
                    nll = F.log_softmax(logits, dim=-1).neg().squeeze()
                    if seq_idx < self.max_len - 1:
                        top_val, top_idx = torch.topk(nll, beam_size, largest=False)
                        for i in range(len(top_idx)):
                            val, idx = top_val[i], top_idx[i]
                            new_seq = torch.cat((seq, idx.view(1, 1)), dim=0).to(X)
                            if idx.item() == EOS_IDX:
                                completed.append((new_seq, score + val))
                            else:
                                new_sequences.append((new_seq, score + val))
                    else:
                        new_seq = torch.cat((seq, torch.LongTensor([[EOS_IDX]]).to(X)), dim=0).to(X)
                        completed.append((new_seq, score + nll[EOS_IDX].item()))
                new_sequences.sort(key=lambda pred: pred[1])
                sequences = new_sequences[:beam_size]

        completed.sort(key=lambda pred: pred[1])
        best, best_score = completed[0]
        return best, best_score
