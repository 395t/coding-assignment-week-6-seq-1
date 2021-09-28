from __future__ import unicode_literals, print_function, division
import os
import json
from torch import optim
from io import open
from model import Encoder, Decoder, Seq2Seq
from train import train_model
from eval import eval_model
from utils import IWSLT2017TransDataset, save, load, makePlot, showTable
from hyper_params import device, hidden_dim, embed_dim, n_layers, dropout, \
    num_epochs, tgt_langs, src_langs, make_plots, evaluate, train

if __name__ == "__main__":

    print("device: {}\n".format(device))

    # Train

    if train:

        for lang_idx in range(0, len(src_langs)):
            # Get current src and tgt language data
            d_train = IWSLT2017TransDataset(src_lang=src_langs[lang_idx], tgt_lang=tgt_langs[lang_idx], dataset_type='train')
            d_valid = IWSLT2017TransDataset(src_lang=src_langs[lang_idx], tgt_lang=tgt_langs[lang_idx], dataset_type='valid')

            encoder = Encoder(source_vocab_size=len(d_train.src_vocab), embed_dim=embed_dim,
                            hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout).to(device)
            decoder = Decoder(target_vocab_size=len(d_train.tgt_vocab), embed_dim=embed_dim,
                            hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout).to(device)

            print("\nTraining {}_{} model\n".format(src_langs[lang_idx], tgt_langs[lang_idx]))
            trained_model, train_losses, valid_losses, optimizer = train_model(encoder, decoder, d_train, d_valid, epochs=num_epochs)

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

    if evaluate:

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

            avg_test_loss, bleu_score = eval_model(model, d, bleu=True)

            avg_test_losses.append(avg_test_loss)
            bleu_scores.append(bleu_score)

            with open(os.path.join(current_model_path, 'states.json'), 'r') as f:
                states_dict = json.load(f)

            states_dict["test_loss"] = avg_test_loss
            states_dict["bleu"] = bleu_score

            with open(os.path.join(current_model_path, 'states.json'), 'w') as f:
                json.dump(states_dict, f)

        # Plot Test losses and bleu

        if make_plots:
            showTable([avg_test_losses], labels, 'Test Set Losses')
            showTable([bleu_scores], labels, 'Test Set Bleu Scores')
