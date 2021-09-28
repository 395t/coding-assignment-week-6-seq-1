from __future__ import unicode_literals, print_function, division
from unicodedata import name
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from io import open
from utils import IWSLT2017TransDataset, save, load
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

src_langs = ['en', 'de', 'it']
tgt_langs = ['de', 'it', 'en']

subset_size = 25000
batch_size = 32
num_epochs = 10
embed_dim = 256
hidden_dim = 512
n_layers = 2
dropout = 0.5

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
        return encoder_out, encoder_hidden

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
        return context, mask

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
        return output, decoder_hidden, mask

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
        decoder_hidden = encoder_hidden[-decoder.n_layers:]  # take what we need from encoder
        output = Variable(torch.zeros(1, batch).long() + self.sos_index)  # start token
        for t in range(self.maxlen):
            output, decoder_hidden, mask = decoder(output, encoder_out, decoder_hidden)
            outputs.append(output)
            masks.append(mask.data)
            output = Variable(output.data.max(dim=2)[1])
        return torch.cat(outputs), torch.cat(masks).permute(1, 2, 0)  # batch, src, trg

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
        return torch.cat(outputs), torch.cat(masks).permute(1, 2, 0)  # batch, src, trg

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, decoding_helper):
        encoder_out, encoder_hidden = self.encoder(source)
        outputs, masks = decoding_helper.generate(self.decoder, encoder_out, encoder_hidden)
        return outputs, masks

############################################ end of model #####################################################

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train_model(encoder, decoder, training_pairs, epochs, lang_idx, print_every=1000, learning_rate=0.01):
    seq2seq = Seq2Seq(encoder, decoder)
    seq2seq.train()
    decoding_helper = Teacher(teacher_forcing_ratio=0.5)

    start = time.time()
    plot_losses = []
    epoch_losses = []
    print_loss_total = 0  # Reset every print_every

    optimizer = optim.SGD(seq2seq.parameters(), lr=learning_rate)

    # compute total number of iterations
    n_iters = math.ceil(subset_size / batch_size)

    for epoch in range(epochs):
        total_loss = 0
        for iter in range(1, n_iters + 1):
            input_batch, target_batch = training_pairs.get_batch(batch_size, iter-1)

            optimizer.zero_grad()

            decoding_helper.load_targets(target_batch)
            outputs, masks = seq2seq(input_batch, decoding_helper)

            loss = F.cross_entropy(outputs.view(-1, outputs.size(2)),
                        target_batch[1:].reshape(-1), ignore_index=1)

            loss.backward()
            nn.utils.clip_grad_norm_(seq2seq.parameters(), 10.0, norm_type=2)  # prevent exploding grads

            optimizer.step()

            print_loss_total += loss
            total_loss += loss.item()

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))
    
        print("Epoch {} finished with avg loss {}".format(epoch, total_loss / n_iters))
        # store average loss for this epoch
        epoch_losses.append(total_loss / n_iters)
    
    # Save loss graph
    showPlot(epoch_losses, lang_idx)

    return seq2seq, epoch_losses, optimizer

def test_model(seq2seq, lang_idx):
    # Get test results
    seq2seq.eval()
    test_d = IWSLT2017TransDataset(src_lang=src_langs[lang_idx], tgt_lang=tgt_langs[lang_idx], dataset_type='valid')
    # Use greedy decoding helper to choose highest score
    decoding_helper_greedy = Greedy()
    decoding_helper_greedy.set_maxlen(49)
    # compute total number of iterations
    n_iters = math.ceil(subset_size / batch_size)
    # evaluate on test set
    total_loss = 0
    for iter in range(1, n_iters + 1):
        input_batch, target_batch = test_d.get_batch(batch_size, iter-1)
        
        outputs, masks = seq2seq(input_batch, decoding_helper_greedy)

        loss = F.cross_entropy(outputs.view(-1, outputs.size(2)),
                           target_batch[1:].reshape(-1), ignore_index=1)

        total_loss += loss.item()

        # preds = outputs.topk(1)[1].squeeze(2)

        # source = test_d.convert_src_ids_to_text(input_batch[:, 0].tolist())
        # prediction = test_d.convert_tgt_ids_to_text(preds[:, 0].tolist())
        # target = test_d.convert_tgt_ids_to_text(target_batch[1:, 0].tolist())

        # print("source: ", source)
        # print("prediction: ", prediction)
        # print("target: ", target)

    avg_valid_loss = total_loss / n_iters
    print("\navg_valid_loss {}\n".format(avg_valid_loss))
    return avg_valid_loss


def showPlot(points, lang_idx):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig("loss_{}_{}.png".format(src_langs[lang_idx], tgt_langs[lang_idx]))


if __name__ == "__main__":

    # Test loading

    # d = IWSLT2017TransDataset(src_lang=src_langs[0], tgt_lang=tgt_langs[0], dataset_type='train')

    # encoder = Encoder(source_vocab_size=len(d.src_vocab), embed_dim=embed_dim,
    #                     hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)
    # decoder = Decoder(target_vocab_size=len(d.tgt_vocab), embed_dim=embed_dim,
    #                 hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)

    # seq2seq = Seq2Seq(encoder, decoder)
    # optimizer = optim.SGD(seq2seq.parameters(), lr=0.1)

    # cwd = os.getcwd()
    # model_path = os.path.join(cwd, 'models')
    # specific_model_path = os.path.join(model_path, '{}_{}'.format(src_langs[0], tgt_langs[0]))

    # epoch, model, optimizer, losses = load(specific_model_path, seq2seq, optimizer)

    # valid_losses = test_model(model, 0)

    # Train and save the three models
    for lang_idx in range(len(src_langs)):
        # Get current src and tgt language data
        d = IWSLT2017TransDataset(src_lang=src_langs[lang_idx], tgt_lang=tgt_langs[lang_idx], dataset_type='train')

        encoder = Encoder(source_vocab_size=len(d.src_vocab), embed_dim=embed_dim,
                        hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)
        decoder = Decoder(target_vocab_size=len(d.tgt_vocab), embed_dim=embed_dim,
                        hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)

        print("\nTraining {}_{} model\n".format(src_langs[lang_idx], tgt_langs[lang_idx]))
        trained_model, train_losses, optimizer = train_model(encoder, decoder, d, epochs=num_epochs, lang_idx=lang_idx)

        # test model on validation set
        print("\nValidating {}_{} model\n".format(src_langs[lang_idx], tgt_langs[lang_idx]))
        valid_losses = test_model(trained_model, lang_idx)

        # Save model
        cwd = os.getcwd()
        model_path = os.path.join(cwd, 'models')
        specific_model_path = os.path.join(model_path, '{}_{}'.format(src_langs[lang_idx], tgt_langs[lang_idx]))
        os.makedirs(specific_model_path, exist_ok=True)

        save(specific_model_path, num_epochs, trained_model, optimizer, train_losses, valid_losses)

    