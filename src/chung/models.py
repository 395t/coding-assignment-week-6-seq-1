import torch
import torch.nn as nn

from utils import PAD_IDX


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
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim=512, rnn_type="gru", hidden_size=None, num_layers=1,
            dropout=0, bidirectional_enc=False, max_len=180):
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
                hidden_size = 558
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
                hidden_size = 695
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
        enc_output, enc_hid = self.enc(self.src_embedding(X))
        if self.bidirectional_enc:
            if self.rnn_type == 'lstm':
                enc_h = self.downsample_h(enc_hid[0])
                enc_c = self.downsample_c(enc_hid[1])
                enc_hid = (enc_h, enc_c)
            else:
                enc_hid = self.downsample_h(enc_hid)
        dec_output, _ = self.dec(self.src_embedding(y_teacher), enc_hid)
        return self.mlp(dec_output)

    def inference(self, X):
        # TODO: beam search
        pass
