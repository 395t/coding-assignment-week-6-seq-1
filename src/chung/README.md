# Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling

Coder: Jay Liao, jl64465

This directory contains the code for the work [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555).

I offer two sets of experiment tasks: music sequence modeling and seq2seq translation. These experiments are meant to test performance differences between GRU, LSTM, and Tanh RNNs. The models in each task have about the same number of parameters.

All code are meant to be run from this directory (training, models) to avoid missing dependencies and other errors.

## Requirements
* Python 3.6.13
* PyTorch 1.9.1
* Torchtext 0.10.1
* Spacy 3.1.3
* Scipy 1.5.2
* Matplotlib 3.3.4
* TorchsummaryX 1.3.0
* tqdm 4.62.3

## Data Prerequisites
```bash
python3 -m spacy download en_core_web_sm
python3 -m spacy download de_core_news_sm
```

You can get the dependencies and data by running `./prereq.sh`. Make sure you have Python 3.6.13 installed.

## Music Sequence Modeling Training
Here's an example training run:
```bash
python3 -m train --task music_lm \
    --rnn gru \
    --data JSB \
    --bs 1 \
    --epoch 500 \
    --lr 0.001 \
    --optim rmsprop \
    --clip 1.0 \
    --loss bce \
    --save_dir music/jsb/gru \
    -v
```
Which corresponds to the following:
* Trains a music LM
* The music LM uses GRU as the underlying RNN
* Train on the JSB dataset
* Batch size of 1 sequence per batch
* Train for 500 epochs
* Use 0.001 for learning rate
* Use the RMSprop optimizer
* Gradient clipping to 1.0
* Use binary cross entropy loss as the training criterion
* Save model to `./exp/music/jsb/gru` and log training to `./log/music/jsb/gru`
* Verbose output (output per epoch loss and per epoch NLL to command line)

For other training arguments, refer to `train.py`.

## Seq2seq Translation Training
TBD

## Data Sources
For the music datasets (JSB, MuseData, Piano-midi, Nottingham), refer to the [TCN repository](https://github.com/locuslab/TCN/tree/master/TCN/poly_music). TCN, or Temporal Convolutional Networks, can be found in the work [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271). You can cite their work with:

```
@article{BaiTCN2018,
	author    = {Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
	title     = {An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling},
	journal   = {arXiv:1803.01271},
	year      = {2018},
}
```
