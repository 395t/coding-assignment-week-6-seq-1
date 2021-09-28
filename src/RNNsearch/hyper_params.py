import torch

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
lr = 0.01

make_plots = False

train = True
evaluate = True