from __future__ import unicode_literals, print_function, division
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from hyper_params import device 

class Encoder(nn.Module):
    def __init__(self, source_vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(source_vocab_size, embed_dim, padding_idx=1)
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, source, hidden=None):
        embedded = self.embed(source)  # (batch_size, seq_len, embed_dim)
        encoder_out, encoder_hidden = self.gru(embedded, hidden)  # (seq_len, batch, hidden_dim*2)
        # sum bidirectional outputs
        encoder_out = (encoder_out[:, :, :self.hidden_dim] +
                       encoder_out[:, :, self.hidden_dim:])
        return encoder_out.to(device), encoder_hidden.to(device)

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.W = nn.Linear(dim, dim, bias=False)

    def score(self, decoder_hidden, encoder_out):
        encoder_out = self.W(encoder_out)
        encoder_out = encoder_out.permute(1, 0, 2)
        return encoder_out @ decoder_hidden.permute(1, 2, 0)

    def forward(self, decoder_hidden, encoder_out):
        energies = self.score(decoder_hidden, encoder_out)
        mask = F.softmax(energies, dim=1)
        context = encoder_out.permute(1, 2, 0) @ mask
        context = context.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)
        return context.to(device), mask.to(device)

class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.embed = nn.Embedding(target_vocab_size, embed_dim, padding_idx=1)
        self.attention = Attention(hidden_dim)
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_dim * 2, target_vocab_size)

    def forward(self, output, encoder_out, decoder_hidden):
        embedded = self.embed(output)  # (1, batch, embed_dim)
        context, mask = self.attention(decoder_hidden[:-1], encoder_out)  # 1, 1, 50 (seq, batch, hidden_dim)
        rnn_output, decoder_hidden = self.gru(torch.cat([embedded, context], dim=2), decoder_hidden)
        output = self.out(torch.cat([rnn_output, context], 2))
        return output.to(device), decoder_hidden.to(device), mask.to(device)

class Greedy:
    def __init__(self, maxlen=20, sos_index=2):
        self.maxlen = maxlen
        self.sos_index = sos_index
        
    def set_maxlen(self, maxlen):
        self.maxlen = maxlen
        
    def generate(self, decoder, encoder_out, encoder_hidden):
        seq, batch, _ = encoder_out.size()
        outputs = []
        masks = []
        decoder_hidden = encoder_hidden[-decoder.n_layers:].to(device)  # take what we need from encoder
        output = Variable(torch.zeros(1, batch).long() + self.sos_index).to(device)  # start token
        for t in range(self.maxlen):
            output, decoder_hidden, mask = decoder(output, encoder_out, decoder_hidden)
            outputs.append(output)
            masks.append(mask.data)
            output = Variable(output.data.max(dim=2)[1])
        return torch.cat(outputs), torch.cat(masks).permute(1, 2, 0).to(device)  # batch, src, trg

class Teacher:
    def __init__(self, teacher_forcing_ratio=0.5):
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.targets = None
        self.maxlen = 0
        
    def load_targets(self, targets):
        self.targets = targets
        self.maxlen = len(targets)

    def generate(self, decoder, encoder_out, encoder_hidden):
        outputs = []
        masks = []
        decoder_hidden = encoder_hidden[-decoder.n_layers:]  # take what we need from encoder
        output = self.targets[0].unsqueeze(0)  # start token
        for t in range(1, self.maxlen):
            output, decoder_hidden, mask = decoder(output, encoder_out, decoder_hidden)
            outputs.append(output)
            masks.append(mask.data)
            output = Variable(output.data.max(dim=2)[1])
            # teacher forcing
            is_teacher = random.random() < self.teacher_forcing_ratio
            if is_teacher:
                output = self.targets[t].unsqueeze(0)      
        return torch.cat(outputs), torch.cat(masks).permute(1, 2, 0).to(device)  # batch, src, trg

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, decoding_helper):
        encoder_out, encoder_hidden = self.encoder(source)
        outputs, masks = decoding_helper.generate(self.decoder, encoder_out, encoder_hidden)
        return outputs.to(device), masks.to(device)