import torch
import torch.nn as nn

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
