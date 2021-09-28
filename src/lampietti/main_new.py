from __future__ import unicode_literals, print_function, division
from unicodedata import name
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import math
import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from io import open
from utils import IWSLT2017TransDataset, save, load
from torch.autograd import Variable
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu

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

make_plots = False

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

def train_model(encoder, decoder, training_pairs, validation_pairs, epochs, lang_idx, print_every=1000, learning_rate=0.01):
    seq2seq = Seq2Seq(encoder, decoder).to(device)
    
    decoding_helper = Teacher(teacher_forcing_ratio=0.5)

    start = time.time()
    plot_losses = []
    epoch_losses = []
    valid_losses = []

    optimizer = optim.SGD(seq2seq.parameters(), lr=learning_rate)

    # compute total number of iterations
    n_iters = math.ceil(subset_size / batch_size)

    for epoch in range(epochs):
        seq2seq.train()
        total_loss = 0
        for iter in range(1, n_iters + 1):
            input_batch, target_batch = training_pairs.get_batch(batch_size, iter-1)
            
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()

            decoding_helper.load_targets(target_batch)
            outputs, masks = seq2seq(input_batch, decoding_helper)

            loss = F.cross_entropy(outputs.view(-1, outputs.size(2)),
                        target_batch[1:].reshape(-1), ignore_index=1)

            loss.backward()
            nn.utils.clip_grad_norm_(seq2seq.parameters(), 10.0, norm_type=2)  # prevent exploding grads

            optimizer.step()

            total_loss += loss.item()
    
        print("Epoch {} finished with avg loss {}".format(epoch, total_loss / n_iters))
        # store average loss for this epoch
        epoch_losses.append(total_loss / n_iters)

        # test on validation set
        valid_loss = test_model(seq2seq, validation_pairs)
        valid_losses.append(valid_loss)

    return seq2seq, epoch_losses, valid_losses, optimizer

def test_model(model, test_pairs, bleu=False):
    # Get test results
    model.eval()
    
    # Use greedy decoding helper to choose highest score
    decoding_helper_greedy = Greedy()
    decoding_helper_greedy.set_maxlen(49)
    # compute total number of iterations
    n_iters = math.ceil(len(test_pairs) / batch_size)
    
    # Track loss and bleu score
    total_loss = 0
    bleu_scores = []
    bleu_score = None

    # evaluate on test set
    for iter in range(1, n_iters + 1):
        input_batch, target_batch = test_pairs.get_batch(batch_size, iter-1)

        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        
        outputs, masks = model(input_batch, decoding_helper_greedy)

        loss = F.cross_entropy(outputs.view(-1, outputs.size(2)),
                           target_batch[1:].reshape(-1), ignore_index=1)

        total_loss += loss.item()

        if bleu:
            preds = outputs.topk(1)[1].squeeze(2)
            # source = test_pairs.convert_src_ids_to_text(input_batch[:, 0].tolist())
            # prediction = test_pairs.convert_tgt_ids_to_text(preds[:, 0].tolist())
            # target = test_pairs.convert_tgt_ids_to_text(target_batch[1:, 0].tolist())
            # print("source: ", source)
            # print("prediction: ", prediction)
            # print("target: ", target)

            for idx in range(0, preds[0].shape[0]):
                ref = target_batch[1:, idx].tolist()
                candidate = preds[:, idx].tolist()
                score = sentence_bleu([ref], candidate)
                bleu_scores.append(score)

    avg_test_loss = total_loss / n_iters
    print("\navg_test_loss {}\n".format(avg_test_loss))
    if bleu:
        bleu_score = sum(bleu_scores) / len(bleu_scores)
        print("bleu score: {}\n".format(bleu_score))
    return avg_test_loss, bleu_score

def makePlot(data, title):

    train_y = data[0]
    valid_y = data[1]

    x = [i for i in range(len(train_y))]

    plt.plot(x, train_y, label = "Train")
    plt.plot(x, valid_y, label = "Valid")
    plt.title('Model {} Loss'.format(title))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def showTable(data, labels, title):
    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=data, colLabels=labels, loc='center')
    plt.title(title)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":

    print("device: {}\n".format(device))

    # Train

    for lang_idx in range(0, len(src_langs)):
        # Get current src and tgt language data
        d_train = IWSLT2017TransDataset(src_lang=src_langs[lang_idx], tgt_lang=tgt_langs[lang_idx], dataset_type='train')
        d_valid = IWSLT2017TransDataset(src_lang=src_langs[lang_idx], tgt_lang=tgt_langs[lang_idx], dataset_type='valid')

        encoder = Encoder(source_vocab_size=len(d_train.src_vocab), embed_dim=embed_dim,
                        hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout).to(device)
        decoder = Decoder(target_vocab_size=len(d_train.tgt_vocab), embed_dim=embed_dim,
                        hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout).to(device)

        print("\nTraining {}_{} model\n".format(src_langs[lang_idx], tgt_langs[lang_idx]))
        trained_model, train_losses, valid_losses, optimizer = train_model(encoder, decoder, d_train, d_valid, epochs=num_epochs, lang_idx=lang_idx)

        # Save model
        cwd = os.getcwd()
        model_path = os.path.join(cwd, 'models')
        current_model_path = os.path.join(model_path, '{}_{}'.format(src_langs[lang_idx], tgt_langs[lang_idx]))
        os.makedirs(current_model_path, exist_ok=True)

        save(current_model_path, num_epochs, trained_model, optimizer, train_losses, valid_losses)

    # Plot training losses

    if make_plots:
        model_names = ['en_de', 'de_it', 'it_en']
        plot_data = {}

        for model in model_names:
            model_dir = os.path.join('models', model)
            with open(os.path.join(model_dir, 'states.json'), 'r') as f:
                states_dict = json.load(f)
            plot_data[model] = (states_dict['train_loss'], states_dict['valid_loss'])

        for model in plot_data:
            makePlot(plot_data[model], model)

    # Evaluate

    avg_test_losses = []
    bleu_scores = []
    labels = []

    for lang_idx in range(0, len(src_langs)):

        # Run trained models on the test sets and store average loss in state dictionary

        d = IWSLT2017TransDataset(src_lang=src_langs[lang_idx], tgt_lang=tgt_langs[lang_idx], dataset_type='test')

        encoder = Encoder(source_vocab_size=len(d.src_vocab), embed_dim=embed_dim,
                            hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout).to(device)
        decoder = Decoder(target_vocab_size=len(d.tgt_vocab), embed_dim=embed_dim,
                        hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout).to(device)

        seq2seq = Seq2Seq(encoder, decoder).to(device)
        optimizer = optim.SGD(seq2seq.parameters(), lr=0.01)

        model_path = os.path.join(os.getcwd(), 'models')
        label = '{}_{}'.format(src_langs[lang_idx], tgt_langs[lang_idx])
        labels.append(label)
        current_model_path = os.path.join(model_path, label)

        epoch, model, optimizer, losses = load(current_model_path, seq2seq, optimizer, device)

        model = model.to(device)

        # Calculate the test set losses and BLEU scores

        avg_test_loss, bleu_score = test_model(model, d, bleu=True)

        avg_test_losses.append(avg_test_loss)
        bleu_scores.append(bleu_score)

        with open(os.path.join(current_model_path, 'states.json'), 'r') as f:
            states_dict = json.load(f)

        states_dict["test_loss"] = avg_test_loss
        states_dict["bleu"] = bleu_score

        with open(os.path.join(current_model_path, 'states.json'), 'w') as f:
            json.dump(states_dict, f)

    showTable([avg_test_losses], labels, 'Test Set Losses')
    showTable([bleu_scores], labels, 'Test Set Bleu Scores')
